"""`full`: re-run experiments from the original input data.

Experiments are listed in experiment_manifest.csv; each runnable one has a configuration in
experiments/configs/<id>.toml whose `kind` selects the runner:

* `m4`     - the M4 campaign runner (campaign.py): workers per arm, checkpoint/resume,
             per-series deadlines; afterwards the run is summarised and, where archived
             per-series results exist, compared series by series with them;
* `script` - a fixed sequence of scripts (the July 2026 battery), each writing its records
             into the run directory with the bundle's layout, so the archived table
             generator rebuilds the tables from the NEW records and compares them with the
             historical ones.

A run never reads archived results in place of its own and never writes into the bundle.
Experiments marked unavailable in the manifest are refused with the reason.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import statistics as st
import sys
import tomllib
from pathlib import Path

from . import campaign, m4data
from .common import (EXIT_FAIL, EXIT_INPUT, EXIT_OK, EXIT_USAGE, FAIL, INFO, MISSING, PASS, WARN,
                     Paths, Report, julia_cmd, run_logged, tail)
from .check import check_m4, compute_fingerprint
from .manifest import load_manifest


def _run_dir(paths: Paths, exp_id: str, run_id: str | None, resume: bool) -> Path:
    base = paths.output / "runs" / exp_id
    if run_id is None:
        if resume and base.exists():
            runs = sorted(p for p in base.iterdir() if p.is_dir())
            if runs:
                return runs[-1]
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = base / run_id
    if d.exists() and not resume:
        raise SystemExit(f"{d} exists; pass --resume to continue it or --run-id to start another run")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------
# Summaries of a campaign run
# ---------------------------------------------------------------------------------------------
def _rows(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["series"]): r for r in csv.DictReader(f)}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def summarise_campaign(paths: Paths, config: dict, run_dir: Path, report: Report, exp_id: str) -> None:
    arms = [a["name"] for a in config["arms"]]
    lines = [f"Run {run_dir.name} of {exp_id}", ""]
    for freq in config["frequencies"]:
        data = {a: _rows(run_dir / f"{a}__{freq}.csv") for a in arms}
        if not any(data.values()):
            continue
        ok = {a: {s: r for s, r in d.items() if r["status"] == "OK"} for a, d in data.items()}
        for a, d in data.items():
            if not d:
                continue
            n_fail = sum(1 for r in d.values() if r["status"] != "OK")
            rate = n_fail / len(d)
            status = PASS if n_fail == 0 else (WARN if rate <= config.get("max_failure_rate", 0.01) else FAIL)
            report.add("full", f"{exp_id} {a}/{freq}: series completed", status,
                       generated=f"{len(d) - n_fail}/{len(d)} OK",
                       note=f"{n_fail} recorded failures (ERROR/TIMEOUT/CRASH rows are kept, never dropped silently)")
        common = None
        for a, d in ok.items():
            if d:
                common = set(d) if common is None else common & set(d)
        if not common:
            continue
        common = sorted(common)
        lines.append(f"{freq} (n={len(common)} series OK in every arm)")
        n2 = "naive2" if ok.get("naive2") else None
        for block in m4data.BLOCK_NAMES:
            parts = []
            if n2:
                bs = st.fmean(_f(ok[n2][s][f"smape_{block}"]) for s in common)
                bm = st.fmean(_f(ok[n2][s][f"mase_{block}"]) for s in common)
            for a in arms:
                if not ok.get(a):
                    continue
                sm = st.fmean(_f(ok[a][s][f"smape_{block}"]) for s in common)
                ma = st.fmean(_f(ok[a][s][f"mase_{block}"]) for s in common)
                owa = f" OWA {0.5 * (ma / bm + sm / bs):.4f}" if n2 else ""
                parts.append(f"{a}: sMAPE {sm:.4f} MASE {ma:.4f}{owa}")
            lines.append(f"  {block:<7} " + " | ".join(parts))
        if not n2:
            lines.append("  (OWA not computed: the naive2 arm was not part of this run)")
        lines.append("")
        # Series-by-series comparison with the archived campaign, where one exists.
        for a in arms:
            ref = config.get("archived", {}).get(a)
            if not ref or not ok.get(a):
                continue
            compare_archived(paths, ref, freq, ok[a], report, f"{exp_id} {a}/{freq}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def compare_archived(paths: Paths, ref: dict, freq: str, rows: dict, report: Report, label: str) -> None:
    from ijftables.common import read_csv
    rel = ref.get("by_frequency", {}).get(freq) or ref["file"].format(freq=freq)
    f = paths.artifacts / rel
    if not f.exists():
        report.add("full", f"{label}: vs archived", INFO, note=f"archived file {rel} not available")
        return
    arch = {int(r[ref.get("series", "series")]): r for r in read_csv(f)}
    same_order = same_metric = n = 0
    for s, r in rows.items():
        a = arch.get(s)
        if a is None or not str(a.get(ref.get("status", "status"), "OK")).strip('"').startswith("OK"):
            continue
        n += 1
        if all(str(r[k]) == str(a.get(k, "")).strip('"') for k in ref.get("order_columns", [])):
            same_order += 1
        sm_new, sm_old = _f(r["smape_total"]), _f(a[ref["smape_column"]])
        if abs(sm_new - sm_old) <= ref.get("tolerance", 1e-4) * max(1.0, abs(sm_old)):
            same_metric += 1
    if n == 0:
        report.add("full", f"{label}: vs archived", INFO, note="no overlapping archived series")
        return
    note = (f"same order {same_order}/{n}; same total sMAPE (rel tol {ref.get('tolerance', 1e-4)}) "
            f"{same_metric}/{n}; archived run: {ref.get('provenance', '')}")
    status = PASS if same_metric == n else WARN
    report.add("full", f"{label}: vs archived per-series results", status,
               expected=f"{n} series", generated=f"{same_metric} reproduce", note=note)


# ---------------------------------------------------------------------------------------------
# Script experiments (the July 2026 battery)
# ---------------------------------------------------------------------------------------------
def run_script_experiment(paths: Paths, config: dict, run_dir: Path, report: Report, exp_id: str) -> bool:
    ok = True
    env = {"IJF_RUN_DIR": str(run_dir), "IJF_FIXTURES": str(paths.repl / "data" / "fixtures"),
           "IJF_PJME_DIR": str(paths.pjme), "IJF_ARTIFACTS_DIR": str(paths.artifacts)}
    py = os.environ.get("IJF_PYTHON", sys.executable)
    for step in config["steps"]:
        script = paths.repl / step["script"]
        cmd = {"julia": julia_cmd(paths), "r": ["Rscript", "--vanilla"], "python": [py]}[step["engine"]] + [str(script)]
        log = run_dir / "logs" / f"{script.stem}.log"
        code, secs = run_logged(cmd + step.get("args", []), log, cwd=paths.repl,
                                timeout=step.get("timeout_s", 7200), env=env)
        status = PASS if code == 0 else FAIL
        ok = ok and code == 0
        report.add("full", f"{exp_id}: {step['script']}", status, generated=f"exit {code} in {secs:.0f}s",
                   note=f"log {log.relative_to(paths.output)}" + ("" if code == 0 else f": {tail(log, 3)}"))
    group = config.get("tables_group")
    if group and ok:
        # Rebuild the tables from the NEW records and compare with the historical ones.
        import importlib
        from ijftables.common import Context
        from .tables import _close, load_expected
        mod = importlib.import_module(f"ijftables.{group}")
        ctx = Context(bundle=run_dir, out=run_dir / "tables", figures=run_dir / "figures")
        # Only the tables of this experiment, from this run's own records.
        cells = {(c.table, c.row, c.column): c for c in mod.generate(ctx, tables=config.get("tables"))}
        expected = [e for e in load_expected(paths, group) if e["table"] in config.get("tables", [e["table"]])]
        agree = 0
        differ = []
        for e in expected:
            c = cells.get((e["table"], e["row"], e["column"]))
            if c is not None and _close(c.value, e["expected"], e["decimals"])[0]:
                agree += 1
            else:
                differ.append(f"{e['table']}[{e['row']}/{e['column']}]")
        report.add("full", f"{exp_id}: rebuilt tables vs historical", PASS if not differ else WARN,
                   expected=f"{len(expected)} cells", generated=f"{agree} agree",
                   difference=f"{len(differ)} differ",
                   note=("re-run on the release environment; differences are expected where the archived "
                         "battery used another package version, solver build or platform. First: "
                         + "; ".join(differ[:5])) if differ else "all cells reproduce")
    return ok


# ---------------------------------------------------------------------------------------------
def all_targets(paths: Paths, manifest: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """`full --all`: the re-runnable experiments behind the manuscript's reported results.

    Experiments the manuscript does not report, historical campaigns superseded by another
    experiment (`superseded_by` in their configuration) and experiments whose protocol is still
    provisional are left out; each can still be run with --experiment.
    """
    targets, left_out = [], []
    for exp_id, e in manifest.items():
        if e["full_runner"] in ("", "none"):
            continue
        cfg = tomllib.loads((paths.repl / e["full_runner"]).read_text(encoding="utf-8"))
        status = cfg.get("protocol_status")
        if status == "provisional":
            left_out.append((exp_id, "its protocol is provisional"))
        elif cfg.get("superseded_by"):
            left_out.append((exp_id, f"superseded by {cfg['superseded_by']}"))
        elif not e["manuscript_label"]:
            left_out.append((exp_id, "not reported in the manuscript"))
        else:
            targets.append(exp_id)
    return targets, left_out


def run(paths: Paths, args) -> int:
    manifest = {e["experiment_id"]: e for e in load_manifest(paths)}
    if args.all:
        targets, left_out = all_targets(paths, manifest)
        for exp_id, why in left_out:
            print(f"--all leaves out {exp_id}: {why} (run it with --experiment {exp_id})")
    else:
        targets = args.experiment
    if not targets:
        print("nothing to run: pass --experiment ID (see `./reproduce.sh list`) or --all")
        return EXIT_USAGE
    unknown = [t for t in targets if t not in manifest]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)} (see `./reproduce.sh list`)")
        return EXIT_USAGE
    report = Report("full", paths)
    source_fp = compute_fingerprint(paths) or os.environ.get("IJF_HOST_FINGERPRINT", "unknown")
    sys.path.insert(0, str(paths.repl / "scripts"))
    status = EXIT_OK
    for exp_id in dict.fromkeys(targets):
        exp = manifest[exp_id]
        if exp["full_runner"] in ("", "none"):
            report.add("full", exp_id, MISSING, note=f"no full re-execution available: {exp['full_status_note']}")
            status = EXIT_FAIL
            continue
        config_path = paths.repl / exp["full_runner"]
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        print(f"\n######## full: {exp_id} ({config['kind']}) ########", flush=True)
        if config["kind"] == "m4":
            freqs = {f for f in config["frequencies"] if not args.frequency or f in args.frequency}
            if not check_m4(paths, report, required=True, frequencies=freqs):
                status = EXIT_INPUT
                if not args.keep_going:
                    break
                continue
        run_dir = _run_dir(paths, exp_id, args.run_id, args.resume)
        events = campaign.Events(run_dir / "events.jsonl")
        events("run_start", experiment=exp_id, limit=args.limit, sample=args.sample, seed=args.seed,
               jobs=args.jobs, resume=args.resume)
        if args.sample:
            report.add("full", f"{exp_id}: sample", WARN,
                       note=f"--sample {args.sample}: {args.sample} series drawn at random per frequency "
                            f"(seed {args.seed}), the same for every arm. This checks the execution path "
                            "on a spread of series; it is NOT a reproduction of the experiment.")
        if args.limit:
            report.add("full", f"{exp_id}: subset", WARN,
                       note=f"--limit {args.limit}: first {args.limit} series per task only. This checks "
                            "the execution path; it is NOT a reproduction of the experiment.")
        try:
            if config["kind"] == "m4":
                campaign.run(paths, exp, config_path, args, run_dir, events, source_fp)
                summarise_campaign(paths, config, run_dir, report, exp_id)
                ok = True
            elif config["kind"] == "script":
                ok = run_script_experiment(paths, config, run_dir, report, exp_id)
            else:
                raise ValueError(f"unknown kind {config['kind']}")
        except (RuntimeError, OSError, ValueError) as exc:
            ok = False
            report.add("full", exp_id, FAIL, note=str(exc))
        events("run_end", experiment=exp_id, ok=ok)
        report.add("full", f"{exp_id}: run directory", INFO, generated=str(run_dir.relative_to(paths.output)))
        if not ok:
            status = EXIT_FAIL
            if not args.keep_going:
                break
    out = report.write()
    print(f"full report: {out}")
    if status == EXIT_OK and report.failed:
        status = EXIT_FAIL
    return status
