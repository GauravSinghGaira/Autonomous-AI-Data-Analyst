"""
Turns the raw profile/anomaly JSON from the backend into plain-English
explanations for the Streamlit UI. Kept separate from streamlit_app.py
so the "explain this number" logic is easy to read and extend on its
own, and so it's pure Python with no Streamlit calls (easy to unit
test if desired).
"""
from __future__ import annotations

from typing import Any


def data_quality_score(profile: dict[str, Any]) -> tuple[int, str]:
    """
    A simple 0-100 quality score derived from missingness and
    duplicate rate, plus a one-line verdict. This is a heuristic,
    not a statistical measure -- it exists to give the user a fast,
    intuitive read on the dataset before they dig into specifics.
    """
    n_rows = max(profile.get("n_rows", 1), 1)
    total_cells = n_rows * max(profile.get("n_cols", 1), 1)
    missing_ratio = profile.get("total_missing_cells", 0) / total_cells if total_cells else 0
    duplicate_ratio = profile.get("duplicate_pct", 0) / 100

    penalty = (missing_ratio * 60) + (duplicate_ratio * 40)
    score = max(0, round(100 - penalty * 100))

    if score >= 90:
        verdict = "Excellent — very clean, ready for analysis as-is."
    elif score >= 75:
        verdict = "Good — minor gaps worth a quick look before modeling."
    elif score >= 50:
        verdict = "Fair — meaningful missing data or duplicates to address first."
    else:
        verdict = "Needs cleanup — significant quality issues before this is analysis-ready."

    return score, verdict


def overview_summary(profile: dict[str, Any]) -> str:
    n_rows = profile.get("n_rows", 0)
    n_cols = profile.get("n_cols", 0)
    numeric_n = len(profile.get("numeric_columns", []))
    cat_n = len(profile.get("categorical_columns", []))
    dt_n = len(profile.get("datetime_columns", []))

    parts = [f"This dataset has **{n_rows:,} rows** and **{n_cols} columns**"]
    type_bits = []
    if numeric_n:
        type_bits.append(f"{numeric_n} numeric")
    if cat_n:
        type_bits.append(f"{cat_n} categorical")
    if dt_n:
        type_bits.append(f"{dt_n} datetime")
    if type_bits:
        parts.append(f"({', '.join(type_bits)})")

    dup_pct = profile.get("duplicate_pct", 0)
    if dup_pct > 0:
        parts.append(f". **{profile.get('duplicate_rows', 0)} rows ({dup_pct}%)** are exact duplicates")
    else:
        parts.append(". No duplicate rows were found")

    missing = profile.get("total_missing_cells", 0)
    if missing > 0:
        parts.append(f", and **{missing} cells** are missing across the dataset")
    else:
        parts.append(", and there are no missing values")

    return "".join(parts) + "."


def column_flags(profile: dict[str, Any], top_n: int = 5) -> list[str]:
    """Flag the columns most worth a user's attention, in plain English."""
    flags = []
    for col in profile.get("columns", []):
        if col["missing_pct"] >= 20:
            flags.append(f"⚠️ **{col['name']}** is {col['missing_pct']}% missing — treat with caution in any aggregation.")
        elif col["group"] == "numeric" and col.get("skew") is not None and abs(col["skew"]) > 2:
            direction = "right" if col["skew"] > 0 else "left"
            flags.append(f"📈 **{col['name']}** is heavily {direction}-skewed — a few extreme values pull the average up/down.")
        elif col["group"] == "categorical" and col["unique_count"] == 1:
            flags.append(f"🔁 **{col['name']}** has only one unique value — it won't be useful for analysis.")
        elif col["group"] != "numeric" and col["unique_count"] == profile.get("n_rows"):
            flags.append(f"🔑 **{col['name']}** has a unique value per row — likely an ID column.")

    return flags[:top_n]


def correlation_narrative(profile: dict[str, Any]) -> str | None:
    pairs = profile.get("correlations", {}).get("strong_pairs", [])
    if not pairs:
        return None
    top = pairs[0]
    direction = "positively" if top["correlation"] > 0 else "negatively"
    strength = "very strongly" if abs(top["correlation"]) > 0.8 else "strongly"
    sentence = (
        f"**{top['col_a']}** and **{top['col_b']}** are {strength} {direction} correlated "
        f"(r = {top['correlation']}) — "
        + ("they tend to move together." if top["correlation"] > 0 else "as one rises, the other tends to fall.")
    )
    if len(pairs) > 1:
        sentence += f" {len(pairs) - 1} other notable correlation(s) were also found."
    return sentence


def anomaly_narrative(anomalies: dict[str, Any]) -> str:
    if "error" in anomalies:
        return anomalies["error"]
    pct = anomalies.get("anomaly_pct", 0)
    n = anomalies.get("n_anomalies", 0)
    cols = ", ".join(anomalies.get("columns_used", []))
    if pct == 0:
        return f"No unusual rows were detected across {cols}. The data looks internally consistent."
    return (
        f"**{n} rows ({pct}%)** look statistically unusual compared to the rest of the dataset, "
        f"based on {cols}. These were flagged by an Isolation Forest model, which isolates points "
        f"that are easy to separate from the bulk of the data — they may be genuine outliers, "
        f"data entry errors, or simply rare-but-valid cases worth a manual look."
    )


SUGGESTED_QUESTIONS = [
    "What's the average of the main numeric column?",
    "Are there any anomalies in this data?",
    "Show me a histogram of the first numeric column",
    "What's the strongest correlation in this dataset?",
    "Can we predict a target column from the others?",
]