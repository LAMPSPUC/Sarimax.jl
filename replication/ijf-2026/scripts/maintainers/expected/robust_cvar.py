"""Expected cells of the `robust_cvar` group.

No generated table of this study was archived: the original analysis scripts printed to the
console only. The reference values below are TRANSCRIBED from the authors' internal notes
(INVENTARIO_IJF.md, section 3.2, dated 2026-08-02), which are not part of the bundle. They
should be replaced by the manuscript's values once the manuscript source is available.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = "transcribed: INVENTARIO_IJF.md s3.2 (authors' internal notes, 2026-08-02; not in the bundle)"

# regime -> (mse, mae, stable alpha=0.9, stable alpha=0.5), as printed (comma decimals in the notes)
LEVEL = {
    "limpo": ("14.846", "15.130", "16.366", "15.304"),
    "1%x5sd": ("15.296", "15.684", "15.409", "15.474"),
    "3%x10sd": ("16.772", "17.294", "18.328", "17.659"),
    "5%x10sd": ("16.899", "17.940", "19.139", "18.444"),
}


def extract(bundle: Path) -> list[dict]:
    rows = [{"table": "robust_cvar.sample", "row": "series", "column": "n", "expected": "200",
             "decimals": "0", "source": SOURCE}]
    for regime, values in LEVEL.items():
        for col, v in zip(("mse", "mae", "stable_a0.9", "stable_a0.5"), values):
            rows.append({"table": "robust_cvar.level", "row": regime, "column": col, "expected": v,
                         "decimals": "3", "source": SOURCE})
    # "Sob 3%x10sd a correcao derrubou o sMAPE de 24,56 -> 18,33"
    rows.append({"table": "robust_cvar.level", "row": "3%x10sd", "column": "stable_prefix",
                 "expected": "24.56", "decimals": "2", "source": SOURCE})
    # "nas vitorias pareadas nenhuma variante cruza 50% em nenhum regime (max: mae 49,0%)"
    rows.append({"table": "robust_cvar.wins_vs_mse", "row": "max over regimes", "column": "mae",
                 "expected": "49.0", "decimals": "1", "source": SOURCE})
    return rows
