"""
Builds a downloadable Markdown analysis report from a dataset's
profile and anomaly results. Kept as plain Markdown (not a PDF
library dependency) so it's simple, diffable, and still renders
nicely if converted to PDF/HTML downstream.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.config import settings


def build_report(filename: str, profile: dict[str, Any], anomalies: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Data Analysis Report — {filename}")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    lines.append("## Dataset Overview")
    lines.append(f"- Rows: **{profile['n_rows']}**")
    lines.append(f"- Columns: **{profile['n_cols']}**")
    lines.append(f"- Duplicate rows: **{profile['duplicate_rows']}** ({profile['duplicate_pct']}%)")
    lines.append(f"- Total missing cells: **{profile['total_missing_cells']}**")
    lines.append(f"- Memory usage: **{profile['memory_usage_kb']} KB**\n")

    lines.append("## Column Summary")
    lines.append("| Column | Type | Missing % | Unique | Notes |")
    lines.append("|---|---|---|---|---|")
    for col in profile["columns"]:
        note = ""
        if col["group"] == "numeric":
            note = f"mean={col.get('mean')}, std={col.get('std')}, range=[{col.get('min')}, {col.get('max')}]"
        elif "top_values" in col:
            top = list(col["top_values"].items())[:3]
            note = ", ".join(f"{k} ({v})" for k, v in top)
        lines.append(f"| {col['name']} | {col['group']} | {col['missing_pct']}% | {col['unique_count']} | {note} |")
    lines.append("")

    strong_pairs = profile.get("correlations", {}).get("strong_pairs", [])
    if strong_pairs:
        lines.append("## Notable Correlations")
        for pair in strong_pairs:
            lines.append(f"- **{pair['col_a']}** ↔ **{pair['col_b']}**: {pair['correlation']}")
        lines.append("")

    lines.append("## Anomaly Detection")
    if "error" in anomalies:
        lines.append(f"_{anomalies['error']}_\n")
    else:
        lines.append(f"- Method: {anomalies['method']}")
        lines.append(f"- Flagged rows: **{anomalies['n_anomalies']}** ({anomalies['anomaly_pct']}%)")
        lines.append(f"- Columns used: {', '.join(anomalies['columns_used'])}\n")

    lines.append("## Notes")
    lines.append(
        "This report was generated automatically from tool-verified computations "
        "(pandas / scikit-learn), not from LLM-estimated figures."
    )

    return "\n".join(lines)


def save_report(filename: str, profile: dict[str, Any], anomalies: dict[str, Any]) -> Path:
    content = build_report(filename, profile, anomalies)
    out_dir = Path(settings.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    out_path = out_dir / f"{stem}_report.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path
