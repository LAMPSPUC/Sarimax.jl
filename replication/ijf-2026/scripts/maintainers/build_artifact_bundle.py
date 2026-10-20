"""Assemble the archived artifact bundle of the IJF replication package (maintainers only).

The bundle is the Zenodo data record that `./reproduce.sh tables` reads. It is assembled
from the places where the experimental material actually lives:

  * this repository's history (the July 2026 battery, removed from master by 74eb882) and
    the three `paper/replication-staging-*` branches;
  * the experiment workspace of the ForecastTester checkout (raw campaign files, the
    auto.arima/Naive2 baselines, the October 2026 Table 7 evidence, the SCIP/Ipopt runs);
  * the `sarimax-replication` folder (the v1.0.0 = a1ebc0d re-run of the huber/ridge cells).

Nothing is recomputed except in one documented transform: the campaign A-F files carry
the M4 test values (`actual`), whose redistribution rights are not established. For those
files the builder computes the per-block sMAPE/MASE from `forecast` and `actual` with the
campaign's own definitions (campaign A's `make_table_m4.jl`), stores them as new columns, and
drops `actual`. Every file's origin, the SHA-256 of the original bytes and the transform
applied are recorded in PROVENANCE.csv, and every file of the bundle in artifacts.sha256.

The output is byte-for-byte deterministic: gzip and tar carry no timestamps or owners.

Usage:
    python build_artifact_bundle.py --repo <Sarimax.jl checkout> \
        --forecasttester <ForecastTester dir> --sarimax-replication <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

BUNDLE = "sarimax-ijf-2026-artifacts"
VERSION = "1"

# Git sources, pinned to the commits bundle v1 was built from (the tips of the
# paper/replication-staging-* branches on 2026-10-17); a moving branch name would let a
# rebuild silently differ. Keep these commits reachable (tag them) before the staging
# branches are deleted.
BATTERY = "308461eef429b8557eece07c50d532ac45d50bea"   # 74eb882^: the last commit tracking the July 2026 battery
ICARAI = "e523509b0038a3c8c30595cf280a3f6a46eb5db9"    # paper/replication-staging-icarai
RECREIO = "052027aa4460ded9c63fd709bacb747715918bb6"   # paper/replication-staging-recreio
GAVEA = "3dae274220183d77ca1c2af92c4dcd7a41768f91"     # paper/replication-staging-gavea

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


@dataclass
class Entry:
    path: str            # path inside the bundle
    data: bytes
    source_kind: str     # git | file
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
    def __init__(self, repo: Path, ft: Path, sr: Path):
        self.repo, self.ft, self.sr = repo, ft, sr
        self.entries: dict[str, Entry] = {}

    # -- sources ---------------------------------------------------------------------------
    def git_files(self, ref: str, prefix: str) -> list[str]:
        out = subprocess.run(["git", "-C", str(self.repo), "ls-tree", "-r", "--name-only", ref, "--", prefix],
                             capture_output=True, check=True, text=True).stdout
        return [p for p in out.splitlines() if p]

    def git_bytes(self, ref: str, path: str) -> bytes:
        return subprocess.run(["git", "-C", str(self.repo), "show", f"{ref}:{path}"],
                              capture_output=True, check=True).stdout

    def add(self, path: str, data: bytes, kind: str, source: str, source_sha: str, transform: str):
        if path in self.entries:
            raise SystemExit(f"duplicate bundle path {path}")
        self.entries[path] = Entry(path, data, kind, source, source_sha, transform)

    def add_git(self, ref: str, src: str, dest: str, compress: bool = False):
        data = self.git_bytes(ref, src)
        out = gz(data) if compress else data
        self.add(dest + (".gz" if compress else ""), out, "git", f"{ref}:{src}", sha256(data),
                 "gzip" if compress else "copy")

    def add_git_tree(self, ref: str, prefix: str, dest: str, compress_csv: bool = False,
                     skip=lambda p: False):
        for p in self.git_files(ref, prefix):
            if skip(p):
                continue
            rel = p[len(prefix):].lstrip("/")
            big = compress_csv and p.endswith((".csv", ".jsonl", ".log")) and not p.endswith(".gz")
            self.add_git(ref, p, f"{dest}/{rel}", compress=big)

    def add_file(self, src: Path, dest: str, compress: bool = False):
        data = src.read_bytes()
        out = gz(data) if compress else data
        self.add(dest + (".gz" if compress else ""), out, "file", self.label(src), sha256(data),
                 "gzip" if compress else "copy")

    def label(self, src: Path) -> str:
        """Workspace-relative name of a local file (no host paths in a public record)."""
        src = src.resolve()
        for root, name in ((self.ft, "ForecastTester"), (self.sr, "sarimax-replication")):
            try:
                return f"{name}/{src.relative_to(root.resolve()).as_posix()}"
            except ValueError:
                continue
        return src.name

    # -- the one transform -----------------------------------------------------------------
    def add_recreio_redacted(self, src: str, dest: str):
        data = self.git_bytes(RECREIO, src)
        text = data.decode("utf-8")
        comments = [l for l in text.splitlines() if l.startswith("#")]
        body = [l for l in text.splitlines() if not l.startswith("#")]
        reader = csv.reader(body)
        header = next(reader)
        if "actual" not in header:
            raise SystemExit(f"{src}: no `actual` column to redact")
        ia, i_f, iden = header.index("actual"), header.index("forecast"), header.index("mase_den")
        ifreq = header.index("freq")
        block_cols = [f"{m}_{b}" for b in ("short", "medium", "long", "total") for m in ("smape", "mase")]
        new_header = [h for h in header if h != "actual"] + block_cols
        out = io.StringIO()
        for c in comments:
            out.write(c + "\n")
        out.write("# redacted for the IJF artifact bundle: column `actual` (M4 test values, whose "
                  "redistribution rights are not established) was removed; the smape_<block>/"
                  "mase_<block> columns were computed from `forecast` and `actual` before removal, "
                  "with the definitions of recreio make_table_m4.jl (see PROVENANCE.csv)\n")
        w = csv.writer(out, lineterminator="\n")
        w.writerow(new_header)
        for row in reader:
            metrics = block_metrics(row[ifreq], row[i_f], row[ia], row[iden])
            w.writerow([v for i, v in enumerate(row) if i != ia] + metrics)
        payload = gz(out.getvalue().encode("utf-8"))
        self.add(dest + ".gz", payload, "git", f"{RECREIO}:{src}", sha256(data),
                 "redact-actual+block-metrics,gzip")

    # -- output ----------------------------------------------------------------------------
    def write(self, out_dir: Path) -> Path:
        root = out_dir / BUNDLE
        if root.exists():
            raise SystemExit(f"{root} exists; remove it first")
        out_dir.mkdir(parents=True, exist_ok=True)
        prov = io.StringIO()
        w = csv.writer(prov, lineterminator="\n")
        w.writerow(["bundle_path", "sha256", "bytes", "source_kind", "source", "source_sha256", "transform"])
        for p in sorted(self.entries):
            e = self.entries[p]
            w.writerow([p, sha256(e.data), len(e.data), e.source_kind, e.source, e.source_sha256, e.transform])
        self.add("PROVENANCE.csv", prov.getvalue().encode("utf-8"), "generated", "build_artifact_bundle.py",
                 "", "generated")
        readme = (Path(__file__).with_name("ARTIFACT_BUNDLE_README.md")).read_bytes()
        self.add("README.md", readme, "generated", "scripts/maintainers/ARTIFACT_BUNDLE_README.md",
                 sha256(readme), "copy")
        for p, e in self.entries.items():
            f = root / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(e.data)
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
                "tarball_bytes": tar_path.stat().st_size}
        (out_dir / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return tar_path


def block_metrics(freq: str, forecast: str, actual: str, den: str) -> list[str]:
    freq = freq.strip('"')
    try:
        f = [float(x) for x in forecast.strip('"').split(";") if x]
        a = [float(x) for x in actual.strip('"').split(";") if x]
        d = float(den)
    except ValueError:
        return ["NaN"] * 8
    out = []
    for block in ("short", "medium", "long", "total"):
        lo, hi = BLOCKS[freq][block]
        if len(f) < hi or len(a) < hi:
            out += ["NaN", "NaN"]
            continue
        fs, as_ = f[lo - 1:hi], a[lo - 1:hi]
        smape = _sum_ltr(2 * abs(x - y) / (abs(x) + abs(y) + 1e-12) for x, y in zip(fs, as_)) / len(fs) * 100
        mase = (_sum_ltr(abs(x - y) for x, y in zip(fs, as_)) / len(fs)) / max(d, 1e-12)
        out += [repr(smape), repr(mase)]
    return out


def _sum_ltr(xs) -> float:
    """Plain left-to-right float sum. The built-in sum() switched to compensated summation in
    Python 3.12, which changes the last bit of some metrics; this loop gives the bytes of the
    v1 bundle (built with Python 3.11) on every Python version."""
    acc = 0.0
    for x in xs:
        acc += x
    return acc


def build(b: Builder) -> None:
    ft, sr = b.ft, b.sr
    # 1. July 2026 battery (validation, forecasting, architecture, solver, energy) ----------
    # PJME data (Kaggle/PJM) is left out: its redistribution terms are not established. The
    # energy runner fetches it from this repository's history at a pinned commit instead.
    pjme = lambda p: "/energy/data/" in p
    b.add_git_tree(BATTERY, "experiments/results/raw", "raw/battery-2026-07", skip=pjme)
    b.add_git_tree(BATTERY, "experiments/tables", "historical-tables/battery-2026-07")
    for p in b.git_files(BATTERY, "experiments"):
        if p.startswith(("experiments/results/", "experiments/tables/")):
            continue
        b.add_git(BATTERY, p, "original-scripts/battery-2026-07/" + p[len("experiments/"):])
    b.add_git(BATTERY, "WORKLOG.md", "original-scripts/battery-2026-07/WORKLOG.md")

    # 2. M4 objective campaigns (August 2026) ------------------------------------------
    pre = "experiments/"
    b.add_git_tree(ICARAI, pre + "results/raw", "raw/m4-objectives-icarai-2026-08/campaigns")
    b.add_git_tree(ICARAI, pre + "results/baselines", "raw/m4-objectives-icarai-2026-08/baselines")
    for name in ("campaigns.csv", "validity.csv", "reproduction_probe_dev.csv", "reproduction_probe_v1.0.0.csv"):
        b.add_git(ICARAI, pre + "results/" + name, "raw/m4-objectives-icarai-2026-08/" + name)
    b.add_git_tree(ICARAI, pre + "tables", "historical-tables/m4-objectives-icarai-2026-08")
    for p in b.git_files(ICARAI, "experiments"):
        if p.startswith(pre + "results/") or p.startswith(pre + "tables/"):
            continue
        b.add_git(ICARAI, p, "original-scripts/m4-objectives-icarai-2026-08/" + p[len(pre):])

    # 3. Re-run of the huber/ridge cells at a1ebc0d (the pre-retag v1.0.0), August 2026 ------
    rr = sr / "runs-v1.0.0"
    for f in sorted(rr.glob("*.csv")):
        b.add_file(f, f"raw/m4-objectives-rerun-a1ebc0d-2026-08/{f.name}", compress=True)
    for f in sorted(rr.glob("*.log")):
        b.add_file(f, f"raw/m4-objectives-rerun-a1ebc0d-2026-08/logs/{f.name}", compress=True)
    for name in ("run_all.sh", "compare_rerun.py"):
        b.add_file(rr / name, f"original-scripts/m4-objectives-rerun-a1ebc0d-2026-08/{name}")

    # 4. Campaigns A-F (August 2026): headline M4 table and the arm experiments ----------
    raw_dir = pre + "results/raw/"
    for p in b.git_files(RECREIO, raw_dir):
        name = p[len(raw_dir):]
        header = b.git_bytes(RECREIO, p).split(b"\n", 4)[3]
        if b"actual" in header.split(b","):
            b.add_recreio_redacted(p, f"raw/m4-recreio-2026-08/{name}")
        else:
            b.add_git(RECREIO, p, f"raw/m4-recreio-2026-08/{name}", compress=True)
    b.add_git_tree(RECREIO, pre + "results/verification", "raw/m4-recreio-2026-08/verification")
    b.add_git_tree(RECREIO, pre + "tables/generated", "historical-tables/m4-recreio-2026-08")
    for p in b.git_files(RECREIO, "experiments"):
        if p.startswith((pre + "results/", pre + "tables/generated/")):
            continue
        b.add_git(RECREIO, p, "original-scripts/m4-recreio-2026-08/" + p[len(pre):])

    # 5. auto.arima and Naive2 per-series baselines, all six M4 frequencies (July 2026) -----
    base = ft / "Results_autoarima_R"
    for freq in ("yearly", "quarterly", "monthly", "weekly", "daily", "hourly"):
        for model in ("autoarima_R", "Naive"):
            for f in sorted((base / freq / model).glob("*.csv")):
                b.add_file(f, f"raw/m4-baselines-r-2026-07/{freq}/{model}/{f.name}", compress=True)
        for f in sorted((base / freq).glob("summary_*.csv")):
            b.add_file(f, f"raw/m4-baselines-r-2026-07/{freq}/{f.name}")
    for name in ("AutoArimaR.jl", "Naive.jl", "../metrics.jl", "../preparedata.jl", "../ForecastTester.jl"):
        src = (ft / "src" / "models" / name).resolve()
        b.add_file(src, f"original-scripts/m4-baselines-r-2026-07/{src.name}")

    # 6. Table 7 distribution evidence (October 2026) -----------------------------------------
    t7 = ft / "table7_results"
    for f in sorted((t7 / "data").iterdir()):
        b.add_file(f, f"derived/m4-objective-distributions-2026-10/{f.name}",
                   compress=f.suffix == ".csv" and f.stat().st_size > 1_000_000)
    for sub in ("tables", "figures"):
        for f in sorted((t7 / sub).iterdir()):
            b.add_file(f, f"historical-tables/m4-objective-distributions-2026-10/{sub}/{f.name}")
    for f in sorted((t7 / "scripts").iterdir()):
        if f.is_file():
            b.add_file(f, f"original-scripts/m4-objective-distributions-2026-10/{f.name}")
    b.add_file(t7 / "table7_evidence.md", "historical-tables/m4-objective-distributions-2026-10/table7_evidence.md")

    # 7. ForecastTester tables and runners behind the August 2026 objective tables -----------
    for name in ("tabelas_paper.txt", "tabela_consolidada.txt"):
        b.add_file(ft / name, f"historical-tables/forecasttester-2026-08/{name}")
    for name in ("tabelas_paper.py", "consolida.py", "compara_objetivos.py", "smoke_innov.jl",
                 "smoke_innov_worker.jl"):
        b.add_file(ft / name, f"original-scripts/forecasttester-2026-08/{name}")

    # 8. Robust objectives (stable/CVaR) under contamination, monthly sample (August 2026) ---
    for name in ("exp_robust_sweep_monthly.csv", "exp_stable_rerun_monthly.csv", "exp_robust_monthly.csv",
                 "smoke_matrix_results.csv", "smoke_matrix_log.txt"):
        b.add_file(ft / name, f"raw/robust-cvar-monthly-2026-08/{name}")
    for name in ("exp_robust.jl", "exp_robust_sweep.jl", "exp_stable_rerun.jl", "analyze_sweep.py",
                 "analyze_stable.py", "smoke_matrix.jl"):
        b.add_file(ft / name, f"original-scripts/robust-cvar-monthly-2026-08/{name}")

    # 9. Admissibility: stationarity/invertibility margins, MA treatment (August 2026) -------
    for name in ("exp_semrestricao.csv", "exp_margem.csv", "exp_rootmargin.csv", "exp_rscheme_ini.csv",
                 "ma_integra.csv", "divergencia_r.csv", "v0_7_monthly.csv", "v0_8_monthly.csv",
                 "piores500.txt", "alvo_ma.txt"):
        b.add_file(ft / name, f"raw/admissibility-2026-08/{name}",
                   compress=(ft / name).stat().st_size > 1_000_000)
    for name in ("exp_semrestricao.log", "exp_margem.log", "exp_rootmargin.log", "exp_rscheme_ini.log",
                 "ma_integra.log"):
        b.add_file(ft / name, f"raw/admissibility-2026-08/logs/{name}", compress=True)
    for name in ("exp_semrestricao.jl", "exp_margem.jl", "exp_rootmargin.jl", "exp_rscheme_ini.jl",
                 "dbg_ma_integra.jl", "analyze_semrestricao.py", "analyze_margem.py", "analyze_ma.py",
                 "analyze_rscheme_ini.py", "analyze_v0_7.py", "analyze_v0_8.py"):
        b.add_file(ft / name, f"original-scripts/admissibility-2026-08/{name}")

    # 10. Initialization and order-search experiments (August 2026) ---------------------------
    for name in ("exp_init_monthly.csv", "exp_init_weekly.csv", "exp_init_daily.csv", "exp_shrinkage_monthly.csv"):
        b.add_file(ft / name, f"raw/initialization-order-search-2026-08/{name}")
    for name in ("exp_init.jl", "analyze_init.py", "exp_shrinkage.jl", "analyze_shrinkage.py"):
        b.add_file(ft / name, f"original-scripts/initialization-order-search-2026-08/{name}")

    # 11. SCIP against Ipopt on short M4 yearly series (July 2026) ----------------------------
    for f in sorted((ft / "docker_scip").iterdir()):
        if f.suffix in (".csv", ".txt"):
            b.add_file(f, f"raw/scip-ipopt-yearly-2026-07/{f.name}")
        elif f.suffix == ".jl":
            b.add_file(f, f"original-scripts/scip-ipopt-yearly-2026-07/{f.name}")

    # 12. Stable/CVaR and admissibility campaigns (September 2026): scripts and probes
    #     only; the raw campaign files travelled in a separate archive not available here.
    for p in b.git_files(GAVEA, "tables"):
        b.add_git(GAVEA, p, f"raw/stable-admissibility-gavea-2026-09/{p}")
    for p in b.git_files(GAVEA, "scripts") + b.git_files(GAVEA, "env") + ["REPRODUCE.md"]:
        b.add_git(GAVEA, p, f"original-scripts/stable-admissibility-gavea-2026-09/{p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--forecasttester", type=Path, required=True)
    ap.add_argument("--sarimax-replication", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    b = Builder(args.repo, args.forecasttester, args.sarimax_replication)
    build(b)
    args.out.mkdir(parents=True, exist_ok=True)
    tar = b.write(args.out)
    print(json.dumps(json.loads((args.out / "bundle.json").read_text()), indent=2))
    print(f"bundle: {tar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
