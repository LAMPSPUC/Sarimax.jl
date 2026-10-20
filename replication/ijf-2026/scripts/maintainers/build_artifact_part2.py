"""Assemble PART 2 of the archived artifact bundle (maintainers only).

Part 1 (`build_artifact_bundle.py`, sarimax-ijf-2026-artifacts-v1) holds the material that
lives in the experiment workspaces and in the repository's staging branches. Part 2 holds the
material produced or received after part 1 was frozen:

  * the controlled comparison of estimation objectives (experiment `m4_objectives`,
    2026-10-17/19): the frozen protocol, the environment, every per-series record of the
    16 (frequency, objective) cells and of the single authorized extra run (weekly MSE under
    the Table 3 protocol), ledgers, completeness audits, event logs, the runner and the
    independent scorer, and the tables that scorer produced;
  * the raw archive `brutos_gavea.zip` (Davi Valladao, shared 2026-10-16): the stable
    (CVaR), over-differencing-guard and 2x2x2x2 admissibility campaigns behind the derived
    tables of 2026-10-17.

The two parts extract into the same directory and are verified together against
artifacts.sha256; part 1 is not modified, so its published SHA-256 stays valid.

Transforms, all recorded per file in part2/PROVENANCE.csv (with the SHA-256 of the original
bytes):

  * M4 test values are not redistributed (their licence is not established; part 1 applies
    the same rule). Controlled-campaign records carry them in the field `test`: the builder
    computes the per-block sMAPE/MASE from `forecast` and `test` (ijflib.m4data, the
    definitions of every M4 runner of this package), stores them as smape_<block> /
    mase_<block>, checks that the runner's own total sMAPE and MASE agree (relative 1e-9),
    and drops `test`. The forecast files of that archive (`*_prev.csv`) carry them in the column `a`,
    which is dropped.
  * Per-series files in the schema of `./reproduce.sh full` (ARTIFACT_GUIDE.md section 7),
    one per (objective, frequency), are derived from the records, so that a new run is
    compared with the archive series by series.
  * Two derived files hold, per series, the ratio between the lag-1 and the seasonal
    in-sample naive MAE of the M4 hourly and quarterly training series: the stable
    campaigns of those frequencies wrote MASE with the lag-1 denominator, and this ratio
    converts it exactly to the M4 definition (reports of 2026-10-17, gavea_archive_audit.md).

The output is byte-for-byte deterministic (gzip and tar carry no timestamps or owners).

Usage:
    python build_artifact_part2.py --repo <Sarimax.jl checkout> \
        --forecasttester <ForecastTester dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from build_artifact_bundle import Builder, gz, sha256  # noqa: E402
from ijflib import m4data  # noqa: E402
from ijflib.campaign import COLUMNS  # noqa: E402

BUNDLE = "sarimax-ijf-2026-artifacts-part2"
VERSION = "1"

CONTROLLED = "m4obj-6957c71f60f1"                       # the 16 cells of the controlled campaign
P1_EXTRA = "m4obj-9f302efccda9-P1-table3-protocol"      # weekly MSE under the Table 3 protocol
FREQS = ("monthly", "quarterly", "yearly", "weekly")
OBJECTIVES = ("mse", "mae", "huber", "ridge")

CC = "raw/m4-objectives-controlled-2026-10"
CCS = "original-scripts/m4-objectives-controlled-2026-10"
CCD = "derived/m4-objectives-controlled-2026-10"
GV = "raw/stable-admissibility-gavea-2026-09/brutos_gavea"
GVD = "derived/stable-admissibility-gavea-2026-09"

# The derived tables of the controlled campaign's independent scorer, as delivered on
# 2026-10-19 (the expected values of the m4_objectives_controlled table group are parsed from
# the .tex/.csv files among them). The per-series score file and the combined ledger are left
# out: the first is recomputed by the table group, the second concatenates the archived ledgers.
DERIVED = ["table5_controlled.csv", "table5_controlled.tex", "table6_controlled.csv",
           "table6_controlled.tex", "table3_weekly_reconciliation.csv",
           "objective_paired_differences.csv", "objective_distribution_quantiles.csv",
           "objective_tail_severity_controlled.csv", "objective_union_tail_controlled.csv",
           "sample_accounting.csv", "reconciliation_checks.csv", "ridge_reconciliation.csv",
           "historical_vs_controlled.csv", "weekly_revision_chains.csv", "guard_domain_decomposition.csv",
           "sensitivity_validity.csv", "validity_flagged_series.csv", "validity_summary.csv",
           "completeness_audit.csv", "figure4_controlled_ecdf.pdf", "figure4_controlled_ecdf.png",
           "reconciliation_report.md", "manuscript_update_proposal.md"]


class Part2(Builder):
    def __init__(self, repo: Path, ft: Path):
        super().__init__(repo, ft, ft)
        self.cc = ft / "paper_results_revision" / "controlled_campaign"
        self.gv = ft / "paper_results_revision" / "_replication_gavea_brutos" / "conteudo"
        self.m4 = ft / "datasets"

    def label(self, src: Path) -> str:
        src = src.resolve()
        try:
            return f"ForecastTester/{src.relative_to(self.ft.resolve()).as_posix()}"
        except ValueError:
            return src.name

    def add_bytes(self, dest: str, data: bytes, source: str, source_sha: str, transform: str,
                  compress: bool = False):
        self.add(dest + (".gz" if compress else ""), gz(data) if compress else data, "file", source,
                 source_sha, transform + (",gzip" if compress else ""))

    # -- output (same tar conventions as part 1; root files under part2/) ----------------------
    def write(self, out_dir: Path) -> Path:
        prov = io.StringIO()
        w = csv.writer(prov, lineterminator="\n")
        w.writerow(["bundle_path", "sha256", "bytes", "source_kind", "source", "source_sha256", "transform"])
        for p in sorted(self.entries):
            e = self.entries[p]
            w.writerow([p, sha256(e.data), len(e.data), e.source_kind, e.source, e.source_sha256, e.transform])
        self.add("part2/PROVENANCE.csv", prov.getvalue().encode("utf-8"), "generated",
                 "build_artifact_part2.py", "", "generated")
        readme = (HERE / "ARTIFACT_PART2_README.md").read_bytes()
        self.add("part2/README.md", readme, "generated", "scripts/maintainers/ARTIFACT_PART2_README.md",
                 sha256(readme), "copy")
        out_dir.mkdir(parents=True, exist_ok=True)
        listing = "".join(f"{sha256(self.entries[p].data)}  {p}\n" for p in sorted(self.entries))
        (out_dir / "artifacts-part2.sha256").write_text(listing, encoding="utf-8", newline="\n")
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
                "inventory_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest()}
        (out_dir / "bundle-part2.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return tar_path


# ---------------------------------------------------------------------------------------------
def m4_rows(m4_dir: Path, freq: str, kind: str) -> list[list[float]]:
    """Rows of an official M4 file; CR is stripped (the files are LF upstream)."""
    stem = m4data.FREQ_FILE[freq]
    text = (m4_dir / f"{stem}-{kind}.csv").read_bytes().replace(b"\r", b"").decode("utf-8")
    out = []
    for row in list(csv.reader(io.StringIO(text)))[1:]:
        out.append([float(x) for x in row[1:] if x.strip() not in ("", "NA")])
    return out


def jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")


def controlled_cell(b: Part2, expid: str, freq: str, obj: str, train: list[list[float]],
                    per_series_dir: str, meta: dict) -> int:
    cell = b.cc / "raw" / expid / freq / obj
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
    dest = f"{CC}/runs/{expid}/{freq}/{obj}"
    b.add_bytes(f"{dest}/records.jsonl", jsonl(redacted), b.label(cell) + "/" + "+".join(s.name for s in shards),
                ";".join(src_sha), "merge-shards,redact-test+block-metrics", compress=True)
    for name in ("ledger.csv", "audit.json"):
        f = cell / name
        b.add_bytes(f"{dest}/{name}", f.read_bytes(), b.label(f), sha256(f.read_bytes()), "copy")
    ev = sorted(cell.glob("events_*.jsonl"))
    evdata = b"".join(e.read_bytes() for e in ev)
    b.add_bytes(f"{dest}/events.jsonl", evdata, b.label(cell) + "/" + "+".join(e.name for e in ev),
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
    b.add_bytes(f"{per_series_dir}/{obj}__{freq}.csv", out.getvalue().encode("utf-8"),
                b.label(cell) + "/records", ";".join(src_sha), "derived:per-series-schema", compress=True)
    return len(final)


def build(b: Part2) -> None:
    # ---- controlled campaign -----------------------------------------------------------------
    train = {f: m4_rows(b.m4, f, "train") for f in FREQS}
    for expid, cells, per_series in ((CONTROLLED, [(f, o) for f in FREQS for o in OBJECTIVES], f"{CC}/per_series"),
                                     (P1_EXTRA, [("weekly", "mse")], f"{CC}/per_series_p1_table3_protocol")):
        meta_path = b.cc / "raw" / expid / "experiment_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        b.add_file(meta_path, f"{CC}/runs/{expid}/experiment_meta.json")
        for f, o in cells:
            n = controlled_cell(b, expid, f, o, train[f], per_series, meta)
            print(f"  {expid} {f}/{o}: {n} series", flush=True)
    for name in ("experiment_protocol.toml", "environment_manifest.json"):
        b.add_file(b.cc / name, f"{CC}/protocol/{name}")
    for name in ("Project.toml", "Manifest.toml"):
        b.add_file(b.cc / "env" / name, f"{CC}/protocol/env/{name}")
    for f in sorted((b.cc / "logs").glob("*.log")):
        b.add_file(f, f"{CC}/logs/{f.name}", compress=True)
    for f in sorted((b.cc / "runner").iterdir()):
        if f.is_file():
            b.add_file(f, f"{CCS}/runner/{f.name}")
    for f in sorted((b.cc / "scoring").glob("*.py")):
        b.add_file(f, f"{CCS}/scoring/{f.name}")
    dl = b.cc / "deliverables" / CONTROLLED
    for name in DERIVED:
        b.add_file(dl / name, f"{CCD}/{name}")

    # ---- raw archive of the September 2026 campaigns -------------------------------------------------------------------
    sums = {}
    for line in (b.gv / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1)
            sums[p.lstrip("*").lstrip("./")] = h
    files = sorted(p for p in b.gv.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(b.gv).as_posix()
        data = f.read_bytes()
        if rel != "SHA256SUMS" and sums.get(rel) not in (None, sha256(data)):
            raise SystemExit(f"archive file {rel}: SHA-256 differs from the archive's SHA256SUMS")
        if rel.endswith("_prev.csv"):
            rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
            if rows[0] != ["sid", "h", "f", "a"]:
                raise SystemExit(f"archive file {rel}: unexpected header {rows[0]}")
            out = io.StringIO()
            cw = csv.writer(out, lineterminator="\n")
            cw.writerows([r[:3] for r in rows])
            b.add_bytes(f"{GV}/{rel}", out.getvalue().encode("utf-8"), b.label(f), sha256(data),
                        "redact-a(M4 test values)", compress=True)
        elif rel == "SHA256SUMS":
            b.add_bytes(f"{GV}/SHA256SUMS.original", data, b.label(f), sha256(data), "copy (sums of the original files)")
        else:
            b.add_file(f, f"{GV}/{rel}", compress=len(data) > 1_000_000)
    # The tables written from this archive on 2026-10-17 (y_updated_results.py, w_gavea_tables.py):
    # the reference values of the table group stable_admissibility_gavea.
    tab = b.ft / "paper_results_revision" / "tables"
    for f in sorted(tab.iterdir()):
        if f.is_file() and (f.name.startswith(("stable_objective_", "admissibility_grid_", "require_terms_guard_", "gavea_"))
                            or f.name == "updated_20261017_captions.tex"):
            b.add_file(f, f"{GVD}/tables-2026-10-17/{f.name}")
    for f in ("y_updated_results.py", "w_gavea_tables.py", "common.py"):
        src = b.ft / "paper_results_revision" / "scripts" / f
        b.add_file(src, f"original-scripts/stable-admissibility-gavea-2026-09/tables-2026-10-17/{f}")
    for freq, s in (("hourly", 24), ("quarterly", 4)):
        out = io.StringIO()
        cw = csv.writer(out, lineterminator="\n")
        cw.writerow(["series", "mase_den_lag1", "mase_den_seasonal", "ratio_lag1_to_seasonal"])
        for i, y in enumerate(m4_rows(b.m4, freq, "train"), start=1):
            d1 = m4data.mase_denominator(y, 1)
            ds = m4data.mase_denominator(y, s)
            cw.writerow([i, repr(d1), repr(ds), repr(d1 / ds)])
        src = b.m4 / f"{m4data.FREQ_FILE[freq]}-train.csv"
        b.add_bytes(f"{GVD}/mase_lag1_to_seasonal_{freq}.csv", out.getvalue().encode("utf-8"), b.label(src),
                    sha256(src.read_bytes().replace(b"\r", b"")), "derived:in-sample-naive-MAE-ratio", compress=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--forecasttester", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    b = Part2(args.repo, args.forecasttester)
    build(b)
    tar = b.write(args.out)
    print(json.dumps(json.loads((args.out / "bundle-part2.json").read_text()), indent=2))
    print(f"bundle part 2: {tar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
