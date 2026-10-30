"""Summary of the default quick reproduction (check + smoke + tables)."""

from __future__ import annotations

import json

from .common import (EXIT_ENV, EXIT_FAIL, EXIT_INPUT, EXIT_OK, Paths, environment_summary, now_iso)

MEANING = {
    EXIT_OK: "PASS",
    EXIT_FAIL: "FAIL",
    EXIT_INPUT: "MISSING INPUT",
    EXIT_ENV: "ENVIRONMENT ERROR",
}


def write_summary(paths: Paths, codes: dict[str, int]) -> int:
    meta = environment_summary(paths)
    lines = ["# SARIMAX.jl IJF replication: quick reproduction summary", "",
             f"- Finished {now_iso()}",
             f"- Image: `{meta['image_ref']}` (identity {meta['image_status']}); runtime {meta['runtime']}",
             f"- Source fingerprint `{meta['source_fingerprint']}`", "",
             "| Stage | Result | Report |", "|---|---|---|"]
    for stage, code in codes.items():
        counts = ""
        try:
            data = json.loads((paths.reports / f"{stage}.json").read_text(encoding="utf-8"))
            counts = ", ".join(f"{k} {v}" for k, v in sorted(data["meta"]["counts"].items()))
        except (OSError, KeyError, ValueError):
            pass
        lines.append(f"| {stage} | **{MEANING.get(code, f'exit {code}')}** ({counts}) | reports/{stage}.md |")
    overall = EXIT_OK
    for code in codes.values():
        if code != EXIT_OK:
            overall = code if overall == EXIT_OK else max(overall, code)
    verdict = "SUCCESS: every check, smoke test and table comparison passed" if overall == EXIT_OK else \
        "FAILURE: see the stage reports; a missing input or a failed check is never counted as a success"
    lines += ["", f"**{verdict}.**"]
    try:
        ms = json.loads((paths.reports / "manuscript.json").read_text(encoding="utf-8"))
        c = ms["counts"]
        lines += ["", f"Manuscript coverage: of {ms['values']} printed values, {c.get('MATCH', 0)} were "
                  f"regenerated from archived per-series results and agree, {c.get('DIFFER', 0)} differ, "
                  f"{c.get('NOT REGENERATED', 0)} could not be regenerated (inputs missing), "
                  f"{c.get('CONSISTENT (derived table only)', 0)} agree with a derived table whose raw inputs "
                  f"are not archived, and {c.get('NO ARCHIVED SOURCE', 0)} have no archived source "
                  "(reports/manuscript_cells.csv; LIMITATIONS.md)."]
    except (OSError, KeyError, ValueError):
        pass
    lines += ["",
              "Full re-execution of the experiments is a separate command: `./reproduce.sh list`, then",
              "`./reproduce.sh full --experiment <id> --jobs N --resume`."]
    paths.reports.mkdir(parents=True, exist_ok=True)
    out = paths.reports / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines[6:]))
    print(f"\nsummary: {out}")
    return overall
