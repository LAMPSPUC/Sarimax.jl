"""Runner of the M4 campaigns (`kind = "m4"` experiments).

For each (frequency, arm) task of an experiment configuration, a pool of worker processes
(experiments/runners/m4_worker.jl for SARIMAX.jl arms, m4_worker.R for the R reference
methods) fits the series one by one. This module owns the data, the metrics, the schedule
and the bookkeeping:

* checkpointing: every finished series is appended to the task's CSV and flushed at once;
* resumption: `--resume` skips the series already in the CSV; a run whose configuration or
  source fingerprint differs from the one it started with is refused;
* per-series deadlines: a series that exceeds the orchestration limit is recorded as
  TIMEOUT, its worker is killed and replaced (a solver stuck in native code cannot be
  interrupted any other way);
* failures: an exception inside a fit is recorded as ERROR:<message>; a worker that dies is
  recorded as CRASH and replaced; nothing is retried silently;
* structured logs: events.jsonl (one JSON object per event) and run.json (configuration
  fingerprint, environment, counts, status).

New results go to <output>/runs/<experiment>/<run id>/ and never touch the archived bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from . import m4data
from .common import Paths, julia_cmd, now_iso, sha256_file

COLUMNS = ["experiment", "arm", "frequency", "series", "T", "status", "p", "d", "q", "P", "D", "Q",
           "has_mean", "has_drift", "solver_status", "fallback", "fit_seconds", "wall_seconds",
           "smape_short", "mase_short", "smape_medium", "mase_medium", "smape_long", "mase_long",
           "smape_total", "mase_total", "mase_den", "forecast", "run_id", "config_sha256",
           "source_fingerprint", "engine"]


class Worker:
    """One worker process speaking the JSON-lines protocol."""

    def __init__(self, cmd: list[str], cwd: Path, log: Path, events, label: str):
        self.label, self.events = label, events
        self.stderr = open(log, "a", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, text=True, encoding="utf-8", bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.job = None              # (series id, start time)
        self.info: dict = {}

    def _read(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)         # end of stream

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                return False
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                self.info = msg
                return True
        return False

    def send(self, job: dict):
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        self.job = (job["series"], time.monotonic())

    def kill(self):
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.stderr.close()


SAMPLE_SEED = 20261018


def sample_ids(ids: list[int], n: int, freq: str, seed: int) -> list[int]:
    """`--sample N`: N series of a frequency drawn at random, the same for every arm.

    The draw is a fixed permutation of the series (seeded by the frequency and the seed, with
    Python's string seeding, stable across platforms and versions) cut at N, so a larger N
    extends a smaller one and `--resume` can grow a sampled run.
    """
    order = list(ids)
    random.Random(f"{freq}|{seed}").shuffle(order)
    return sorted(order[:n])


class Events:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, event: str, **detail):
        self.f.write(json.dumps({"time": now_iso(), "event": event, **detail}) + "\n")
        self.f.flush()


def config_fingerprint(config_path: Path) -> str:
    return sha256_file(config_path, strip_cr=True)


def worker_command(paths: Paths, config_path: Path, arm: dict) -> list[str]:
    runners = paths.repl / "experiments" / "runners"
    if arm["engine"] == "julia":
        return julia_cmd(paths) + [str(runners / "m4_worker.jl"), str(config_path), arm["name"]]
    if arm["engine"] == "r":
        return ["Rscript", "--vanilla", str(runners / "m4_worker.R"), arm["method"]]
    raise ValueError(f"unknown engine {arm['engine']}")


def read_done(path: Path, events=None) -> set[int]:
    """Series already in a task's checkpoint file.

    A row cut short by an interruption (wrong number of fields, no status) is not a finished
    series: the file is rewritten without it (the original is kept as *.bak) and the series
    is fitted again.
    """
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != COLUMNS:
        raise RuntimeError(f"{path} does not have the runner's header; refusing to resume into it")
    good = [r for r in rows[1:] if len(r) == len(COLUMNS) and r[COLUMNS.index("status")]]
    if len(good) != len(rows) - 1:
        os.replace(path, path.with_suffix(".csv.bak"))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(COLUMNS)
            w.writerows(good)
        if events:
            events("checkpoint_repaired", file=path.name, dropped_rows=len(rows) - 1 - len(good))
    return {int(r[COLUMNS.index("series")]) for r in good}


def run_task(paths: Paths, exp: dict, config_path: Path, arm: dict, freq: str, run_dir: Path,
             ids: list[int], train, test, jobs: int, timeout: float, events: Events,
             meta: dict) -> dict:
    """Fit `ids` with `jobs` workers; returns counts. Appends to <arm>__<freq>.csv."""
    out_csv = run_dir / f"{arm['name']}__{freq}.csv"
    done = read_done(out_csv, events)
    pending = [i for i in ids if i not in done]
    counts = {"requested": len(ids), "already_done": len(ids) - len(pending), "ok": 0, "error": 0,
              "timeout": 0, "crash": 0}
    if not pending:
        return counts
    new_file = not out_csv.exists()
    fh = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, COLUMNS, lineterminator="\n", extrasaction="ignore")
    if new_file:
        writer.writeheader()
        fh.flush()
    s, H = m4data.SEASONALITY[freq], m4data.HORIZON[freq]
    cmd = worker_command(paths, config_path, arm)
    label = f"{arm['name']}/{freq}"
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)

    def spawn(k: int) -> Worker | None:
        for attempt in range(3):
            w = Worker(cmd, paths.repl, logs / f"{arm['name']}__{freq}__worker{k}.log", events, label)
            if w.wait_ready(timeout=900):
                events("worker_ready", task=label, worker=k, info=w.info)
                return w
            events("worker_failed_to_start", task=label, worker=k, attempt=attempt + 1)
            w.kill()
        return None

    def write(series: int, res: dict, wall: float):
        y, a = train[series - 1], test[series - 1]
        row = {k: "" for k in COLUMNS}
        row.update(experiment=exp["experiment_id"], arm=arm["name"], frequency=freq, series=series, T=len(y),
                   status=res.get("status", ""), run_id=meta["run_id"], config_sha256=meta["config_sha256"],
                   source_fingerprint=meta["source_fingerprint"], wall_seconds=round(wall, 3),
                   engine=meta["engines"].get(arm["name"], ""))
        for k in ("p", "d", "q", "P", "D", "Q", "has_mean", "has_drift", "solver_status", "fallback",
                  "fit_seconds"):
            if k in res:
                row[k] = res[k]
        if row["status"] == "OK":
            f = [float(x) for x in res["forecast"]]
            met = m4data.block_metrics(freq, y, a, f)
            row.update({k: repr(v) for k, v in met.items()})
            row["forecast"] = ";".join(repr(x) for x in f)
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())
        st = row["status"]
        key = "ok" if st == "OK" else "timeout" if st == "TIMEOUT" else "crash" if st.startswith("CRASH") else "error"
        counts[key] += 1

    workers: dict[int, Worker] = {}
    for k in range(max(1, min(jobs, len(pending)))):
        w = spawn(k)
        if w is None:
            fh.close()
            raise RuntimeError(f"{label}: worker could not start (see {logs})")
        workers[k] = w
        meta["engines"][arm["name"]] = json.dumps(w.info, sort_keys=True)
    queue_ids = list(pending)
    t_task = time.monotonic()
    restarts = 0
    try:
        while queue_ids or any(w.job for w in workers.values()):
            for k, w in list(workers.items()):
                if w.job is None and queue_ids:
                    sid = queue_ids[0]
                    try:
                        w.send({"series": sid, "y": train[sid - 1], "s": s, "H": H})
                    except OSError:                  # the worker died while idle: replace it
                        events("worker_died_idle", task=label, worker=k, exit=w.proc.poll())
                        w.kill()
                        restarts += 1
                        workers[k] = spawn(k) or _fail(label, logs)
                        continue
                    queue_ids.pop(0)
            progressed = False
            for k, w in list(workers.items()):
                if w.job is None:
                    continue
                sid, t0 = w.job
                try:
                    line = w.lines.get_nowait()
                except queue.Empty:
                    line = ""
                if line is None:                                    # the process died
                    write(sid, {"status": f"CRASH:exit {w.proc.poll()}"}, time.monotonic() - t0)
                    events("worker_crash", task=label, worker=k, series=sid, exit=w.proc.poll())
                    w.kill()
                    restarts += 1
                    workers[k] = spawn(k) or _fail(label, logs)
                    progressed = True
                elif line:
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if res.get("series") != sid:
                        continue
                    write(sid, res, time.monotonic() - t0)
                    w.job = None
                    progressed = True
                elif time.monotonic() - t0 > timeout:
                    write(sid, {"status": "TIMEOUT"}, time.monotonic() - t0)
                    events("timeout", task=label, worker=k, series=sid, limit_s=timeout)
                    w.kill()
                    restarts += 1
                    workers[k] = spawn(k) or _fail(label, logs)
                    progressed = True
            if restarts > max(50, len(pending) // 10):
                raise RuntimeError(f"{label}: {restarts} worker restarts; aborting the task")
            n = counts["ok"] + counts["error"] + counts["timeout"] + counts["crash"]
            if progressed and (n % 25 == 0 or n == len(pending)):
                el = time.monotonic() - t_task
                print(f"    {label}: {n}/{len(pending)} done ({counts['error']} errors, "
                      f"{counts['timeout']} timeouts, {counts['crash']} crashes) in {el:.0f}s", flush=True)
            if not progressed:
                time.sleep(0.05)
    finally:
        for w in workers.values():
            if w is not None:
                try:
                    w.proc.stdin.close()
                except OSError:
                    pass
                w.kill()
        fh.close()
    return counts


def _fail(label: str, logs: Path):
    raise RuntimeError(f"{label}: replacement worker could not start (see {logs})")


def run(paths: Paths, exp: dict, config_path: Path, args, run_dir: Path, events: Events,
        source_fp: str) -> dict:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    freqs = [f for f in config["frequencies"] if not args.frequency or f in args.frequency]
    arms = [a for a in config["arms"] if not args.arm or a["name"] in args.arm]
    timeout = args.timeout or float(config["per_series_timeout_s"])
    meta_path = run_dir / "run.json"
    meta = {"experiment": exp["experiment_id"], "run_id": run_dir.name, "config": str(config_path.relative_to(paths.repl)),
            "config_sha256": config_fingerprint(config_path), "source_fingerprint": source_fp,
            "image": os.environ.get("IJF_IMAGE_REF", "(native)"), "protocol_status": config.get("protocol_status"),
            "frequencies": freqs, "arms": [a["name"] for a in arms], "limit": args.limit,
            "sample": getattr(args, "sample", 0), "seed": getattr(args, "seed", SAMPLE_SEED),
            "jobs": args.jobs, "per_series_timeout_s": timeout, "engines": {}, "tasks": {}}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "source_fingerprint"):
            if old.get(key) != meta[key]:
                raise SystemExit(f"refusing to resume {run_dir}: {key} changed "
                                 f"({old.get(key)} -> {meta[key]}); start a new --run-id")
        meta["tasks"] = old.get("tasks", {})
        meta["engines"] = old.get("engines", {})
    meta["started"] = meta.get("started") or now_iso()
    if config.get("protocol_status") == "provisional":
        print(f"  NOTE: {exp['id']} uses a PROVISIONAL protocol: {config.get('protocol_note', '')}", flush=True)
    for freq in freqs:
        train, test = m4data.load(paths.m4, freq)
        ids = list(range(1, len(train) + 1))
        if args.limit:
            ids = ids[:args.limit]
        if getattr(args, "sample", 0):
            ids = sample_ids(ids, args.sample, freq, getattr(args, "seed", SAMPLE_SEED))
        for arm in arms:
            label = f"{arm['name']}__{freq}"
            print(f"  task {label}: {len(ids)} series, {args.jobs} workers, limit {timeout:.0f}s/series", flush=True)
            events("task_start", task=label, series=len(ids))
            counts = run_task(paths, exp, config_path, arm, freq, run_dir, ids, train, test,
                              args.jobs, timeout, events, meta)
            out_csv = run_dir / f"{label}.csv"
            if out_csv.exists():          # integrity record of the task's output as written
                counts["rows_in_file"] = len(read_done(out_csv, events))
                counts["sha256"] = sha256_file(out_csv)
            meta["tasks"][label] = counts
            events("task_end", task=label, **counts)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["finished"] = now_iso()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
