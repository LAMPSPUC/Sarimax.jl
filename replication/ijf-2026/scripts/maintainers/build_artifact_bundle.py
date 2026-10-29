"""Assemble the archived artifact bundle of the IJF replication package (maintainers only).

The bundle is the Zenodo data record that `./reproduce.sh tables` reads. It holds the
archived material behind the values printed in the submitted manuscript and nothing else:
for each experiment, the raw results and run logs, the intermediate files the table groups
read, the tables as they were generated at the time, and the runners and environment pins
that produced them. Every file is selected explicitly below; nothing is copied by directory.

Sources (all on the maintainers' workspace):

  * this repository's history: the July 2026 battery (commit BATTERY), the August 2026 M4
    campaigns A-F (commit BENCHMARK) and the scripts of the September 2026 stable and
    admissibility campaigns (commit SEPTEMBER);
  * the ForecastTester workspace: the auto.arima/Naive2 per-series baselines and the M4
    harness (July 2026), the controlled objective campaign (October 2026), the raw archive
    of the September 2026 campaigns and the tables written from it on 2026-10-17.

Transforms, recorded per file in PROVENANCE.csv with the SHA-256 of the original bytes:

  * M4 test values are not redistributed (their licence is not established). Campaign A-F
    files carry them in the column `actual`: the per-block sMAPE/MASE are computed from
    `forecast` and `actual` with the campaign's own definitions (make_table_m4.jl), stored as
    smape_<block> / mase_<block>, and `actual` is dropped. Controlled-campaign records carry
    them in the field `test`: the per-block metrics are computed (ijflib.m4data), checked
    against the runner's own total sMAPE and MASE (relative 1e-9), and `test` is dropped.
    The forecast files of the September archive (`*_prev.csv`) carry them in the column `a`,
    which is dropped.
  * Per-series files in the schema of `./reproduce.sh full` (ARTIFACT_GUIDE.md section 7),
    one per (objective, frequency) of the controlled campaign, are derived from its records.
  * Two derived files hold, per series, the ratio between the lag-1 and the seasonal
    in-sample naive MAE of the M4 hourly and quarterly training series: the stable
    campaigns of those frequencies wrote MASE with the lag-1 denominator, and this ratio
    converts it exactly to the M4 definition.

The output is byte-for-byte deterministic: gzip and tar carry no timestamps or owners.

Usage:
    python build_artifact_bundle.py --repo <Sarimax.jl checkout> \
        --forecasttester <ForecastTester dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ijflib import m4data  # noqa: E402
from ijflib.campaign import COLUMNS  # noqa: E402

BUNDLE = "sarimax-ijf-2026-artifacts"
VERSION = "1"

# Git sources, pinned by commit; a moving branch name would let a rebuild silently differ.
# Keep these commits reachable (tag them) before the branches that hold them are deleted.
BATTERY = "308461eef429b8557eece07c50d532ac45d50bea"    # 74eb882^: the last commit tracking the July 2026 battery
BENCHMARK = "052027aa4460ded9c63fd709bacb747715918bb6"  # campaigns A-F (August 2026): raw results, runners, tables
SEPTEMBER = "3dae274220183d77ca1c2af92c4dcd7a41768f91"  # runners and environment of the September 2026 campaigns

# Bundle directories, one per experiment.
BAT = "battery-2026-07"
BENCH = "m4-benchmark-2026-08"
BASE = "m4-baselines-2026-07"
CTRL = "m4-objectives-controlled-2026-10"
STAB = "stable-admissibility-2026-09"

FREQS6 = ("yearly", "quarterly", "monthly", "weekly", "daily", "hourly")
BLOCK_NAMES = ("short", "medium", "long", "total")

# Horizon blocks of the M4 harness (ForecastTester src/ForecastTester.jl), shared by every
# campaign and by the stored auto.arima/Naive2 baselines.
BLOCKS = {
    "monthly": {"short": (1, 6), "medium": (7, 12), "long": (13, 18), "total": (1, 18)},
    "quarterly": {"short": (1, 2), "medium": (3, 5), "long": (6, 8), "total": (1, 8)},
    "yearly": {"short": (1, 2), "medium": (3, 4), "long": (5, 6), "total": (1, 6)},
    "weekly": {"short": (1, 4), "medium": (5, 9), "long": (10, 13), "total": (1, 13)},
    "daily": {"short": (1, 4), "medium": (5, 9), "long": (10, 14), "total": (1, 14)},
    "hourly": {"short": (1, 16), "medium": (17, 32), "long": (33, 48), "total": (1, 48)},
}

# -- selection -------------------------------------------------------------------------------
# July 2026 battery: the blocks behind Table 2 (validation), the MA(1) certificates of
# Section 5.6 (solver comparison) and Table 7 (local against global estimation).
BATTERY_RAW = [
    "validation/julia_results.jsonl", "validation/python_statsmodels_results.jsonl",
    "validation/r_forecast_results.jsonl", "validation/data/sim_arma.csv", "validation/data/sim_sarimax.csv",
    "solver/global_value_results.jsonl", "solver/julia_results.jsonl", "solver/jump_results.jsonl",
    "solver/scip_cert_results.jsonl", "solver/global_value_run.log", "solver/jump_run.log",
    "solver/package_run.log", "solver/refresh.log", "solver/refresh_scip.log",
    "environment/git_commit.txt", "environment/julia_status.txt", "environment/julia_version.txt",
    "environment/os_uname.txt", "environment/python_freeze.txt", "environment/python_versions.json",
    "environment/python_versions.txt", "environment/r_version.txt", "environment/run_timestamp.txt",
]
BATTERY_TABLES = [f"table_{t}.{ext}" for t in ("validation", "solver_comparison", "global_value")
                  for ext in ("csv", "tex")]
BATTERY_SCRIPTS = [
    "env/Project.toml", "env/Manifest.toml", "requirements.txt", "SHA256SUMS",
    "scripts/bench_common.jl", "scripts/combine_results.py",
    "scripts/run_validation_julia.jl", "scripts/run_validation_python.py", "scripts/run_validation_r.R",
    "scripts/run_solver_diagnostics.jl", "scripts/run_solver_jump_diagnostics.jl",
    "scripts/run_solver_scip_cert.jl", "scripts/run_solver_global_value.jl",
    "scripts/setup/gen_sim_data.py", "scripts/setup/gen_energy_data.py",
    "scripts/setup/install_r.R", "scripts/setup/julia_setup.jl",
]
# Campaign A (Table 3, six frequencies) and campaign F (stable objective, weekly and yearly).
BENCH_RAW = [f"m4_innovations_{f}.csv" for f in FREQS6] + ["stable_weekly_yearly.csv"]
BENCH_SCRIPTS = [
    "env/Project.toml", "env/Manifest.toml", "env/Manifest-moi131.toml",
    "scripts/provenance.jl", "scripts/run_m4.jl", "scripts/run_stable.jl", "scripts/backfill_provenance.jl",
    "tables/make_table_m4.jl",
]
# auto.arima and Naive2: the four horizon blocks of every frequency, and the harness.
BASE_MODELS = ("autoarima_R", "Naive")
BASE_SCRIPTS = ["src/models/AutoArimaR.jl", "src/models/Naive.jl", "src/metrics.jl", "src/preparedata.jl",
                "src/ForecastTester.jl"]
# Controlled objective campaign (Tables 5, 6, S13, S14, Figure 3 and the controlled column of
# Table S4): the 16 cells and the weekly MSE run under the Table 3 protocol.
CONTROLLED = "m4obj-6957c71f60f1"
P1_EXTRA = "m4obj-9f302efccda9-P1-table3-protocol"
FREQS4 = ("monthly", "quarterly", "yearly", "weekly")
OBJECTIVES = ("mse", "mae", "huber", "ridge")
CTRL_TABLES = ["table5_controlled.csv", "table5_controlled.tex", "table6_controlled.csv",
               "table6_controlled.tex", "objective_paired_differences.csv",
               "objective_distribution_quantiles.csv"]
# The weekly MSE comparison of Table S4 and Section 5.1: the rows of the six-frequency
# campaign, the controlled cell and the controlled-environment run under the Table 3 protocol.
WEEKLY_TABLE = "table3_weekly_reconciliation.csv"
WEEKLY_ROWS = ("Table 3 published (P1, aa68d57, archived)", "controlled P2 (14dd581)",
               "controlled-environment P1 extra run (14dd581)")
CTRL_SCRIPTS = ["runner/environment_manifest.jl", "runner/run_all.sh", "runner/run_campaign.jl",
                "runner/worker.jl", "scoring/score_campaign.py"]
# September 2026 campaigns (Tables S15 and S16, Section 5.5): the stable objective on the
# monthly, quarterly, daily and hourly series and the 2x2x2x2 admissibility grid on monthly.
STAB_ARCHIVE_DIRS = {"stable_monthly": "stable_monthly", "stable_quarterly": "stable_quarterly",
                     "stable_daily": "stable_daily", "stable_hourly": "stable_hourly",
                     "censo_2x2x2x2": "admissibility_grid"}
GRID_CELLS = ("gs", "gff", "gi", "gsi", "fs", "f1", "fi", "fsi")
STAB_TABLES = [f"{s}_{k}.csv" for s in ("stable_objective", "admissibility_grid")
               for k in ("summary", "paired_differences")]
STAB_SCRIPTS = ["env/Project.toml", "scripts/campanha_stable_runner.jl", "scripts/campanha_stable_fila.sh",
                "scripts/runner2.jl", "scripts/supervisor.sh"]


@dataclass
class Entry:
    path: str            # path inside the bundle
    data: bytes
    source_kind: str     # git | file | generated
    source: str
    source_sha256: str
    transform: str


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gz(b: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as f:
        f.write(b)
    return buf.getvalue()


class Builder:
    def __init__(self, repo: Path, ft: Path):
        self.repo, self.ft = repo, ft
        self.entries: dict[str, Entry] = {}

    # -- sources ---------------------------------------------------------------------------
    def git_bytes(self, ref: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(self.repo), "show", f"{ref}:{path}"],
                              capture_output=True, check=True).stdout

    def label(self, src: Path) -> str:
        """Workspace-relative name of a local file (no host paths in a public record)."""
        try:
            return f"ForecastTester/{src.resolve().relative_to(self.ft.resolve()).as_posix()}"
        except ValueError:
            return src.name

    def add(self, path: str, data: bytes, kind: str, source: str, source_sha: str, transform: str):
        if path in self.entries:
            raise SystemExit(f"duplicate bundle path {path}")
        self.entries[path] = Entry(path, data, kind, source, source_sha, transform)

    def add_bytes(self, dest: str, data: bytes, kind: str, source: str, source_sha: str, transform: str,
                  compress: bool = False):
        self.add(dest + (".gz" if compress else ""), gz(data) if compress else data, kind, source,
                 source_sha, transform + (",gzip" if compress else ""))

    def add_git(self, ref: str, src: str, dest: str, compress: bool = False):
        data = self.git_bytes(ref, src)
        self.add(dest + (".gz" if compress else ""), gz(data) if compress else data, "git", f"{ref}:{src}",
                 sha256(data), "gzip" if compress else "copy")

    def add_file(self, src: Path, dest: str, compress: bool = False):
        data = src.read_bytes()
        self.add(dest + (".gz" if compress else ""), gz(data) if compress else data, "file", self.label(src),
                 sha256(data), "gzip" if compress else "copy")

    # -- output ----------------------------------------------------------------------------
    def write(self, out_dir: Path) -> Path:
        if (out_dir / BUNDLE).exists():
            raise SystemExit(f"{out_dir / BUNDLE} exists; remove it first")
        out_dir.mkdir(parents=True, exist_ok=True)
        prov = io.StringIO()
        w = csv.writer(prov, lineterminator="\n")
        w.writerow(["bundle_path", "sha256", "bytes", "source_kind", "source", "source_sha256", "transform"])
        for p in sorted(self.entries):
            e = self.entries[p]
            w.writerow([p, sha256(e.data), len(e.data), e.source_kind, e.source, e.source_sha256, e.transform])
        self.add("PROVENANCE.csv", prov.getvalue().encode("utf-8"), "generated", "build_artifact_bundle.py",
                 "", "generated")
        readme = (HERE / "ARTIFACT_BUNDLE_README.md").read_bytes()
        self.add("README.md", readme, "generated", "scripts/maintainers/ARTIFACT_BUNDLE_README.md",
                 sha256(readme), "copy")
        listing = "".join(f"{sha256(self.entries[p].data)}  {p}\n" for p in sorted(self.entries))
        (out_dir / "artifacts.sha256").write_text(listing, encoding="utf-8", newline="\n")
        tar_path = out_dir / f"{BUNDLE}-v{VERSION}.tar.gz"
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(self.entries):
                data = self.entries[p].data
                info = tarfile.TarInfo(f"{BUNDLE}/{p}")
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
        tar_path.write_bytes(gz(raw.getvalue()))
        meta = {"bundle": BUNDLE, "version": VERSION, "files": len(self.entries),
                "bytes": sum(len(e.data) for e in self.entries.values()),
                "tarball": tar_path.name, "tarball_sha256": sha256(tar_path.read_bytes()),
                "tarball_bytes": tar_path.stat().st_size,
                "inventory_sha256": sha256(listing.encode("utf-8"))}
        (out_dir / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return tar_path


# ---------------------------------------------------------------------------------------------
# Campaigns A-F: the M4 test values are replaced by per-block metrics
# ---------------------------------------------------------------------------------------------
def _sum_ltr(xs) -> float:
    """Plain left-to-right float sum. The built-in sum() switched to compensated summation in
    Python 3.12, which changes the last bit of some metrics; this loop gives the same bytes on
    every Python version."""
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def block_metrics(freq: str, forecast: str, actual: str, den: str) -> list[str]:
    freq = freq.strip('"')
    try:
        f = [float(x) for x in forecast.strip('"').split(";") if x]
        a = [float(x) for x in actual.strip('"').split(";") if x]
        d = float(den)
    except ValueError:
        return ["NaN"] * 8
    out = []
    for block in BLOCK_NAMES:
        lo, hi = BLOCKS[freq][block]
        if len(f) < hi or len(a) < hi:
            out += ["NaN", "NaN"]
            continue
        fs, as_ = f[lo - 1:hi], a[lo - 1:hi]
        smape = _sum_ltr(2 * abs(x - y) / (abs(x) + abs(y) + 1e-12) for x, y in zip(fs, as_)) / len(fs) * 100
        mase = (_sum_ltr(abs(x - y) for x, y in zip(fs, as_)) / len(fs)) / max(d, 1e-12)
        out += [repr(smape), repr(mase)]
    return out


def add_benchmark_redacted(b: Builder, src: str, dest: str):
    data = b.git_bytes(BENCHMARK, src)
    text = data.decode("utf-8")
    comments = [l for l in text.splitlines() if l.startswith("#")]
    body = [l for l in text.splitlines() if not l.startswith("#")]
    reader = csv.reader(body)
    header = next(reader)
    if "actual" not in header:
        raise SystemExit(f"{src}: no `actual` column to remove")
    ia, i_f, iden = header.index("actual"), header.index("forecast"), header.index("mase_den")
    ifreq = header.index("freq")
    block_cols = [f"{m}_{blk}" for blk in BLOCK_NAMES for m in ("smape", "mase")]
    out = io.StringIO()
    for c in comments:
        out.write(c + "\n")
    out.write("# column `actual` (M4 test values, whose redistribution rights are not established) was "
              "removed; the smape_<block>/mase_<block> columns were computed from `forecast` and `actual` "
              "before removal, with the definitions of make_table_m4.jl (see PROVENANCE.csv)\n")
    if any("REPRODUCE.md" in c for c in comments):
        out.write("# REPRODUCE.md, cited above, is a note of the campaign's working directory and is not part "
                  "of this record; the recorded and reconstructed versions are given per experiment in "
                  "experiment_manifest.csv of the replication package\n")
    w = csv.writer(out, lineterminator="\n")
    w.writerow([h for h in header if h != "actual"] + block_cols)
    for row in reader:
        metrics = block_metrics(row[ifreq], row[i_f], row[ia], row[iden])
        w.writerow([v for i, v in enumerate(row) if i != ia] + metrics)
    b.add(dest + ".gz", gz(out.getvalue().encode("utf-8")), "git", f"{BENCHMARK}:{src}", sha256(data),
          "remove-actual+block-metrics,gzip")


# ---------------------------------------------------------------------------------------------
# Controlled campaign: records without the M4 test values, and per-series files
# ---------------------------------------------------------------------------------------------
def m4_rows(m4_dir: Path, freq: str, kind: str) -> list[list[float]]:
    """Rows of an official M4 file; CR is stripped (the files are LF upstream)."""
    stem = m4data.FREQ_FILE[freq]
    text = (m4_dir / f"{stem}-{kind}.csv").read_bytes().replace(b"\r", b"").decode("utf-8")
    return [[float(x) for x in row[1:] if x.strip() not in ("", "NA")]
            for row in list(csv.reader(io.StringIO(text)))[1:]]


def jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")


def controlled_cell(b: Builder, cc: Path, expid: str, freq: str, obj: str, train: list[list[float]],
                    per_series_dir: str, meta: dict) -> int:
    cell = cc / "raw" / expid / freq / obj
    shards = sorted(cell.glob("records_*.jsonl"))
    if not shards:
        raise SystemExit(f"{cell}: no records")
    records, src_sha = [], []
    for s in shards:
        data = s.read_bytes()
        src_sha.append(f"{s.name}:{sha256(data)}")
        records += [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]
    records.sort(key=lambda r: (int(r["sid"]), int(r.get("attempt", 1))))
    redacted, final = [], {}
    for r in records:
        r = dict(r)
        test = r.pop("test", None)
        if r.get("outcome") in ("ok", "ok_solver_status_other") and test is not None:
            y = train[int(r["sid"]) - 1]
            f = [float(x) for x in r["forecast"]]
            met = m4data.block_metrics(freq, y, [float(x) for x in test], f)
            for k in ("smape", "mase"):
                own = float(r[k])
                if not math.isclose(own, met[f"{k}_total"], rel_tol=1e-9, abs_tol=1e-12):
                    raise SystemExit(f"{expid} {freq}/{obj} series {r['sid']}: runner {k} {own} "
                                     f"differs from {met[f'{k}_total']}")
            if not math.isclose(float(r["mase_denominator"]), met["mase_den"], rel_tol=1e-9):
                raise SystemExit(f"{expid} {freq}/{obj} series {r['sid']}: MASE denominator differs")
            r.update({k: v for k, v in met.items() if k != "mase_den"})
        redacted.append(r)
        final[int(r["sid"])] = r
    dest = f"raw/{CTRL}/runs/{expid}/{freq}/{obj}"
    b.add_bytes(f"{dest}/records.jsonl", jsonl(redacted), "file",
                b.label(cell) + "/" + "+".join(s.name for s in shards), ";".join(src_sha),
                "merge-shards,remove-test+block-metrics", compress=True)
    for name in ("ledger.csv", "audit.json"):
        b.add_file(cell / name, f"{dest}/{name}")
    ev = sorted(cell.glob("events_*.jsonl"))
    evdata = b"".join(e.read_bytes() for e in ev)
    b.add_bytes(f"{dest}/events.jsonl", evdata, "file", b.label(cell) + "/" + "+".join(e.name for e in ev),
                sha256(evdata), "concatenate-shards", compress=True)

    # Per-series file in the schema of `./reproduce.sh full` (one row per series, final attempt).
    out = io.StringIO()
    w = csv.DictWriter(out, COLUMNS, lineterminator="\n")
    w.writeheader()
    for sid in sorted(final):
        r = final[sid]
        ok = r.get("outcome") in ("ok", "ok_solver_status_other")
        order = r.get("order") or [""] * 7
        row = {k: "" for k in COLUMNS}
        row.update(experiment="m4_objectives", arm=obj, frequency=freq, series=sid, T=r.get("T", ""),
                   status="OK" if ok else str(r.get("outcome", "")).upper(),
                   run_id=expid, config_sha256=meta["protocol_sha256"], source_fingerprint=meta["fingerprint"],
                   engine="julia; controlled_campaign/runner/worker.jl (archived run)")
        if ok:
            c, tr = r.get("c"), r.get("trend")
            row.update(p=order[0], d=order[1], q=order[2], P=order[3], D=order[4], Q=order[5],
                       has_mean=int(c is not None and abs(float(c)) > 1e-10),
                       has_drift=int(tr is not None and abs(float(tr)) > 1e-10),
                       solver_status=r.get("solver_status", ""),
                       fallback=(int(bool(r["huber_fallback"])) if r.get("huber_fallback") is not None else -1),
                       fit_seconds=repr(float(r["fit_seconds"])),
                       mase_den=repr(float(r["mase_denominator"])),
                       forecast=";".join(repr(float(x)) for x in r["forecast"]))
            row.update({f"{m}_{blk}": repr(r[f"{m}_{blk}"]) for blk in m4data.BLOCK_NAMES for m in ("smape", "mase")})
        w.writerow(row)
    b.add_bytes(f"{per_series_dir}/{obj}__{freq}.csv", out.getvalue().encode("utf-8"), "file",
                b.label(cell) + "/records", ";".join(src_sha), "derived:per-series-schema", compress=True)
    return len(final)


# ---------------------------------------------------------------------------------------------
def build(b: Builder) -> None:
    ft = b.ft
    m4_dir = ft / "datasets"

    # 1. July 2026 battery -------------------------------------------------------------------
    for p in BATTERY_RAW:
        b.add_git(BATTERY, f"experiments/results/raw/{p}", f"raw/{BAT}/{p}")
    for p in BATTERY_TABLES:
        b.add_git(BATTERY, f"experiments/tables/{p}", f"historical-tables/{BAT}/{p}")
    for p in BATTERY_SCRIPTS:
        b.add_git(BATTERY, f"experiments/{p}", f"original-scripts/{BAT}/{p}")

    # 2. Campaigns A and F (August 2026) -------------------------------------------------------
    for name in BENCH_RAW:
        add_benchmark_redacted(b, f"experiments/results/raw/{name}", f"raw/{BENCH}/{name}")
    b.add_git(BENCHMARK, "experiments/tables/generated/table_m4.txt", f"historical-tables/{BENCH}/table_m4.txt")
    for p in BENCH_SCRIPTS:
        b.add_git(BENCHMARK, f"experiments/{p}", f"original-scripts/{BENCH}/{p}")

    # 3. auto.arima and Naive2 baselines (July 2026) -------------------------------------------
    for freq in FREQS6:
        for model in BASE_MODELS:
            for blk in BLOCK_NAMES:
                b.add_file(ft / "Results_autoarima_R" / freq / model / f"{blk}.csv",
                           f"raw/{BASE}/{freq}/{model}/{blk}.csv", compress=True)
    for p in BASE_SCRIPTS:
        b.add_file(ft / p, f"original-scripts/{BASE}/{Path(p).name}")

    # 4. Controlled objective campaign (October 2026) -------------------------------------------
    cc = ft / "paper_results_revision" / "controlled_campaign"
    train = {f: m4_rows(m4_dir, f, "train") for f in FREQS4}
    for expid, cells, per_series in ((CONTROLLED, [(f, o) for f in FREQS4 for o in OBJECTIVES],
                                      f"raw/{CTRL}/per_series"),
                                     (P1_EXTRA, [("weekly", "mse")], f"raw/{CTRL}/per_series_p1_table3_protocol")):
        meta_path = cc / "raw" / expid / "experiment_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        b.add_file(meta_path, f"raw/{CTRL}/runs/{expid}/experiment_meta.json")
        for f, o in cells:
            n = controlled_cell(b, cc, expid, f, o, train[f], per_series, meta)
            print(f"  {expid} {f}/{o}: {n} series", flush=True)
    for name in ("experiment_protocol.toml", "environment_manifest.json"):
        b.add_file(cc / name, f"raw/{CTRL}/protocol/{name}")
    for name in ("Project.toml", "Manifest.toml"):
        b.add_file(cc / "env" / name, f"raw/{CTRL}/protocol/env/{name}")
    for f in sorted((cc / "logs").glob("campaign_*.log")):
        b.add_file(f, f"raw/{CTRL}/logs/{f.name}", compress=True)
    for name in CTRL_TABLES:
        b.add_file(cc / "deliverables" / CONTROLLED / name, f"historical-tables/{CTRL}/{name}")
    src = cc / "deliverables" / CONTROLLED / WEEKLY_TABLE
    data = src.read_bytes()
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    kept = [rows[0]] + [r for r in rows[1:] if r[0] in WEEKLY_ROWS]
    if len(kept) != len(WEEKLY_ROWS) + 1:
        raise SystemExit(f"{src}: expected the rows {WEEKLY_ROWS}")
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(kept)
    b.add_bytes(f"historical-tables/{CTRL}/{WEEKLY_TABLE}", out.getvalue().encode("utf-8"), "file", b.label(src),
                sha256(data), "select-rows(six-frequency campaign, controlled cell, Table 3 protocol run)")
    for p in CTRL_SCRIPTS:
        b.add_file(cc / p, f"original-scripts/{CTRL}/{p}")

    # 5. Stable objective and admissibility grid (September 2026) -------------------------------
    gv = ft / "paper_results_revision" / "_replication_gavea_brutos" / "conteudo"
    sums = {}
    for line in (gv / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1)
            sums[p.lstrip("*").lstrip("./")] = h
    for src_dir, dest_dir in STAB_ARCHIVE_DIRS.items():
        for f in sorted((gv / src_dir).iterdir()):
            name = f.name
            if src_dir == "censo_2x2x2x2" and not name.startswith(tuple(f"{c}_" for c in GRID_CELLS)):
                continue      # the stable monthly files, also copied into this folder, are kept once
            data = f.read_bytes()
            rel = f"{src_dir}/{name}"
            if sums.get(rel) != sha256(data):
                raise SystemExit(f"archive file {rel}: SHA-256 differs from the archive's SHA256SUMS")
            dest = f"raw/{STAB}/{dest_dir}/{name}"
            if name.endswith("_prev.csv"):
                rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
                if rows[0] != ["sid", "h", "f", "a"]:
                    raise SystemExit(f"archive file {rel}: unexpected header {rows[0]}")
                out = io.StringIO()
                csv.writer(out, lineterminator="\n").writerows([r[:3] for r in rows])
                b.add_bytes(dest, out.getvalue().encode("utf-8"), "file", b.label(f), sha256(data),
                            "remove-a(M4 test values)", compress=True)
            else:
                b.add_file(f, dest, compress=len(data) > 1_000_000)
    for freq, s in (("hourly", 24), ("quarterly", 4)):
        out = io.StringIO()
        cw = csv.writer(out, lineterminator="\n")
        cw.writerow(["series", "mase_den_lag1", "mase_den_seasonal", "ratio_lag1_to_seasonal"])
        for i, y in enumerate(m4_rows(m4_dir, freq, "train"), start=1):
            d1, ds = m4data.mase_denominator(y, 1), m4data.mase_denominator(y, s)
            cw.writerow([i, repr(d1), repr(ds), repr(d1 / ds)])
        src = m4_dir / f"{m4data.FREQ_FILE[freq]}-train.csv"
        b.add_bytes(f"derived/{STAB}/mase_lag1_to_seasonal_{freq}.csv", out.getvalue().encode("utf-8"), "file",
                    b.label(src), sha256(src.read_bytes().replace(b"\r", b"")),
                    "derived:in-sample-naive-MAE-ratio", compress=True)
    for name in STAB_TABLES:
        b.add_file(ft / "paper_results_revision" / "tables" / name, f"historical-tables/{STAB}/{name}")
    for p in STAB_SCRIPTS:
        b.add_git(SEPTEMBER, p, f"original-scripts/{STAB}/{p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--forecasttester", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    b = Builder(args.repo, args.forecasttester)
    build(b)
    tar = b.write(args.out)
    print(json.dumps(json.loads((args.out / "bundle.json").read_text()), indent=2))
    print(f"bundle: {tar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
