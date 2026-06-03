"""Write collected rows to CSV / XLSX in the output dir and return a download ref.

The agent gathers rows during a run (the `record_rows` action) and writes a file
with the `export` action. xlsx needs openpyxl; if it's missing we fall back to csv
so a run never loses its data.
"""
from __future__ import annotations

import csv
import re

import config


def _safe_name(name: str, default: str = "export") -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip("/. ")).strip("_")
    return (base or default)[:80]


def _columns(rows: list[dict], columns) -> list[str]:
    if columns:
        return [str(c) for c in columns]
    seen: list[str] = []
    for r in rows:
        for k in r.keys():
            if str(k) not in seen:
                seen.append(str(k))
    return seen or ["value"]


def _cell(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def write_table(rows: list[dict], filename: str, fmt: str = "xlsx", columns=None) -> dict:
    """Write `rows` to output/<name>.<fmt>. Returns {filename, rows, columns, url, format}."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    fmt = (fmt or "xlsx").lower()
    if fmt not in ("xlsx", "csv"):
        fmt = "xlsx"
    cols = _columns(rows, columns)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_name(filename)

    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError:
            fmt = "csv"  # graceful fallback so data is never lost

    path = config.OUTPUT_DIR / f"{name}.{fmt}"

    if fmt == "csv":
        # utf-8-sig so Excel opens non-ASCII correctly.
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    else:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(cols)
        for r in rows:
            ws.append([_cell(r.get(c, "")) for c in cols])
        wb.save(path)

    return {
        "filename": path.name,
        "rows": len(rows),
        "columns": cols,
        "url": f"/output/{path.name}",
        "format": fmt,
    }
