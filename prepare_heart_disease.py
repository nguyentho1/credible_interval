"""Prepare the four processed UCI heart-disease databases for analysis.

Usage:
    python scripts/prepare_heart_disease.py

The source files are deliberately kept unchanged. Missing values are converted
to pandas NA and a binary target is derived from the original ``num`` target.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DATA = ROOT / "data"
OUTPUT_REPORTS = ROOT / "reports"

SOURCES = {
    "cleveland": ROOT / "processed.cleveland.data",
    "hungary": ROOT / "processed.hungarian.data",
    "switzerland": ROOT / "processed.switzerland.data",
    "va_long_beach": ROOT / "processed.va.data",
}

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

EXPECTED_ROWS = {
    "cleveland": 303,
    "hungary": 294,
    "switzerland": 123,
    "va_long_beach": 200,
}

MISSING_MARKERS = ["?", "-9", "-9.0"]


def load_source(name: str, path: Path) -> pd.DataFrame:
    """Read and validate one 14-column processed source file."""
    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=MISSING_MARKERS,
        keep_default_na=True,
        skipinitialspace=True,
    )
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{name} has {frame.shape[1]} columns; expected {len(COLUMNS)}"
        )
    expected_rows = EXPECTED_ROWS[name]
    if len(frame) != expected_rows:
        raise ValueError(f"{name} has {len(frame)} rows; expected {expected_rows}")
    if frame["num"].isna().any():
        raise ValueError(f"{name} contains missing target values")

    frame["source"] = name
    frame["target_binary"] = (frame["num"] > 0).astype("int8")
    return frame


def build_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize row counts, missingness, and target distributions by source."""
    rows = []
    for source, group in frame.groupby("source", sort=False):
        for column in COLUMNS:
            rows.append(
                {
                    "source": source,
                    "column": column,
                    "rows": len(group),
                    "missing": int(group[column].isna().sum()),
                    "missing_rate": float(group[column].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DATA.mkdir(exist_ok=True)
    OUTPUT_REPORTS.mkdir(exist_ok=True)

    frames = [load_source(name, path) for name, path in SOURCES.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_DATA / "heart_disease_combined.csv", index=False)

    quality = build_quality_summary(combined)
    quality.to_csv(OUTPUT_REPORTS / "data_quality_summary.csv", index=False)

    class_distribution = (
        combined.groupby(["source", "num", "target_binary"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    class_distribution.to_csv(
        OUTPUT_REPORTS / "class_distribution.csv", index=False
    )

    print(f"Prepared {len(combined)} rows from {combined['source'].nunique()} sources")
    print(f"Wrote {OUTPUT_DATA / 'heart_disease_combined.csv'}")
    print(f"Wrote {OUTPUT_REPORTS / 'data_quality_summary.csv'}")
    print(f"Wrote {OUTPUT_REPORTS / 'class_distribution.csv'}")


if __name__ == "__main__":
    main()
