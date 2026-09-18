# snippets.py
# General-purpose Pandas / NumPy / Matplotlib toolkit
# Focus: practical Data Analyst drills and reusable syntax.
#
# Workflow:
# LOAD -> INSPECT -> DATA QUALITY -> FILTER/SORT -> GROUPBY -> STATS
# -> OUTLIER -> CORRELATION -> VISUALIZATION

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_csv(path, **kwargs):
    """Load a CSV file."""
    return pd.read_csv(path, **kwargs)


def load_excel(path, sheet_name=0, **kwargs):
    """Load an Excel file."""
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


# ============================================================
# 2. BASIC INSPECTION
# ============================================================

def peek(df, n=5):
    """Show shape, columns, dtypes, head, and missing values."""
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print(f"\nFirst {n} rows:")
    display(df.head(n))
    print("\nMissing values:")
    display(df.isna().sum().sort_values(ascending=False))


def columns(df):
    """Return column names."""
    return df.columns.tolist()


def numeric_cols(df):
    """Return numeric column names."""
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_cols(df):
    """Return categorical/object column names."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def describe_all(df):
    """Descriptive statistics for all columns."""
    return df.describe(include="all").T


# ============================================================
# 3. DATA QUALITY
# ============================================================

def missing_summary(df, sort=True):
    """Missing count and percentage per column."""
    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean().mul(100).round(2)
    })
    if sort:
        result = result.sort_values("missing_count", ascending=False)
    return result


def duplicates_summary(df, subset=None):
    """Return duplicate-row count and percentage."""
    count = df.duplicated(subset=subset).sum()
    pct = count / len(df) * 100 if len(df) else 0
    return pd.Series({
        "duplicate_count": count,
        "duplicate_pct": round(pct, 2)
    })


def unique_summary(df):
    """Number of unique values in every column."""
    return df.nunique(dropna=False).sort_values(ascending=False)


def freq(df, col, normalize=False, top_n=None):
    """Frequency table for a categorical column."""
    result = df[col].value_counts(
        normalize=normalize,
        dropna=False
    )
    if normalize:
        result = (result * 100).round(2)
    if top_n is not None:
        result = result.head(top_n)
    return result


# ============================================================
# 4. COLUMN CLEANING — USE ONLY WHEN NEEDED
# ============================================================

def clean_columns(df):
    """Standardize column names to snake_case."""
    df = df.copy()

    def to_snake(name):
        name = str(name).strip()
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_").lower()

    df.columns = [to_snake(c) for c in df.columns]
    return df

def spaces_to_underscore(df):
    """Replace spaces in column names with underscores."""
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_")
    return df

def strip_text(df):
    """Remove leading/trailing whitespace from text columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def to_datetime(df, col, errors="coerce"):
    """Convert a column to datetime."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors=errors)
    return df


# ============================================================
# 5. FILTER & SORT
# ============================================================

def filter_rows(df, condition):
    """Generic row filtering. Example: filter_rows(df, df['Age'] > 30)."""
    return df.loc[condition]


def top_n(df, col, n=10, ascending=False):
    """Return top/bottom N rows based on a numeric column."""
    return df.sort_values(col, ascending=ascending).head(n)


def sort_by(df, col, ascending=True):
    """Sort dataframe by one column."""
    return df.sort_values(col, ascending=ascending)


# ============================================================
# 6. GROUPBY & AGGREGATION
# ============================================================

def group_stats(df, group_col, value_col, stats=None):
    """
    Grouped statistics.
    Default: count, mean, median, min, max.
    """
    if stats is None:
        stats = ["count", "mean", "median", "min", "max"]

    result = df.groupby(group_col)[value_col].agg(stats)

    if "mean" in result.columns:
        result = result.sort_values("mean", ascending=False)

    return result


def group_count(df, group_col, sort=True):
    """Count rows by category."""
    result = df.groupby(group_col).size().rename("count")
    return result.sort_values(ascending=False) if sort else result


def group_mean(df, group_col, value_col):
    """Mean of a numeric variable by category."""
    return df.groupby(group_col)[value_col].mean().sort_values(ascending=False)


def crosstab_count(df, row_col, col_col):
    """Count cross-tabulation."""
    return pd.crosstab(df[row_col], df[col_col])


def crosstab_pct(df, row_col, col_col):
    """Row-percentage cross-tabulation."""
    return pd.crosstab(
        df[row_col],
        df[col_col],
        normalize="index"
    ).mul(100).round(2)


# ============================================================
# 7. STATISTICS
# ============================================================

def describe_col(df, col):
    """Quick statistics for one column."""
    return df[col].describe()


def correlation(df, method="pearson"):
    """Correlation matrix for numeric columns."""
    return df.select_dtypes(include=np.number).corr(method=method)


def corr_pairs(df, min_abs=0.0):
    """
    Return unique correlation pairs sorted by absolute correlation.
    Self-correlations and duplicate pairs are removed.
    """
    corr = correlation(df).abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)

    pairs = (
        corr.where(mask)
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    return pairs[pairs["abs_corr"] >= min_abs].sort_values(
        "abs_corr", ascending=False
    )


# ============================================================
# 8. OUTLIER — IQR METHOD
# ============================================================

def outlier_iqr(df, col):
    """
    Detect outliers using the IQR rule.
    Returns Q1, Q3, IQR, lower/upper bounds, count and percentage.
    """
    s = df[col].dropna()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (df[col] < lower) | (df[col] > upper)
    count = mask.sum()

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count,
        "outlier_pct": round(count / len(df) * 100, 2) if len(df) else 0,
        "outlier_rows": df.loc[mask]
    }


# ============================================================
# 9. VISUALIZATION
# ============================================================

def plot_count(df, col, top_n=None, title=None):
    """Bar chart of category counts."""
    data = df[col].value_counts(dropna=False)

    if top_n is not None:
        data = data.head(top_n)

    data.plot(kind="bar", figsize=(9, 5))
    plt.title(title or f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_hist(df, col, bins=30, title=None):
    """Histogram for a numeric column."""
    df[col].dropna().plot(
        kind="hist",
        bins=bins,
        figsize=(9, 5)
    )
    plt.title(title or f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_box(df, col, by=None, title=None):
    """Boxplot for one numeric column, optionally grouped."""
    plt.figure(figsize=(9, 5))

    if by is None:
        plt.boxplot(df[col].dropna())
        plt.ylabel(col)
    else:
        df.boxplot(column=col, by=by, figsize=(9, 5))
        plt.suptitle("")
        plt.xlabel(by)
        plt.ylabel(col)

    plt.title(title or f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


def plot_bar(df, x, y, title=None, ascending=False, top_n=None):
    """Bar chart from two columns."""
    data = df[[x, y]].dropna().sort_values(y, ascending=ascending)

    if top_n is not None:
        data = data.head(top_n)

    plt.figure(figsize=(9, 5))
    plt.bar(data[x].astype(str), data[y])
    plt.title(title or f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x, y, title=None):
    """Scatter plot."""
    data = df[[x, y]].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(data[x], data[y], alpha=0.5)
    plt.title(title or f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_line(df, x, y, title=None):
    """Simple line chart."""
    data = df[[x, y]].dropna().sort_values(x)

    plt.figure(figsize=(9, 5))
    plt.plot(data[x], data[y])
    plt.title(title or f"{y} over {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, figsize=(10, 8), title="Correlation Heatmap"):
    """Correlation heatmap using Matplotlib only."""
    corr = correlation(df)

    plt.figure(figsize=figsize)
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. QUICK EDA
# ============================================================

def quick_eda(df):
    """
    Fast dataset overview.
    IMPORTANT: this only inspects the data; it does not clean or delete rows.
    """
    print("=" * 60)
    print("QUICK EDA")
    print("=" * 60)

    print("\nShape:", df.shape)

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nMissing values:")
    display(missing_summary(df).head(15))

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumeric columns:")
    print(numeric_cols(df))

    print("\nCategorical columns:")
    print(categorical_cols(df))

    print("\nPreview:")
    display(df.head())


# ============================================================
# 11. COMMON RAW PANDAS — MEMORIZE THESE
# ============================================================

# df.head()
# df.tail()
# df.shape
# df.info()
# df.describe()
# df.columns
# df.dtypes
# df.isna().sum()
# df.duplicated().sum()
# df["Column"].unique()
# df["Column"].nunique()
# df["Column"].value_counts()
# df["Column"].value_counts(normalize=True) * 100
#
# df[df["Age"] > 30]
# df[df["Category"] == "A"]
# df.sort_values("Sales", ascending=False)
#
# df.groupby("Category")["Sales"].mean()
# df.groupby("Category")["Sales"].sum()
# df.groupby("Category").size()
#
# pd.crosstab(df["Category"], df["Label"])
#
# df["Date"] = pd.to_datetime(df["Date"])
#
# df.corr(numeric_only=True)


# ============================================================
# 12. QUICK CHEAT SHEET
# ============================================================

# INSPECTION
# peek(df)
# df.shape
# df.info()
# df.describe()
#
# QUALITY
# missing_summary(df)
# duplicates_summary(df)
# unique_summary(df)
#
# CATEGORY
# freq(df, "Label")
# freq(df, "Label", normalize=True)
# top_n(freq(df, "Label").reset_index(), "count", 5)
#
# GROUPBY
# group_count(df, "Label")
# group_mean(df, "Label", "Flow Duration")
# group_stats(df, "Label", "Flow Duration")
#
# FILTER
# filter_rows(df, df["Age"] > 30)
# top_n(df, "Sales", 10)
#
# OUTLIER
# outlier_iqr(df, "Total Fwd Packets")
#
# CORRELATION
# correlation(df)
# corr_pairs(df, min_abs=0.7)
#
# VISUALIZATION
# plot_count(df, "Label")
# plot_hist(df, "Age")
# plot_box(df, "Income")
# plot_scatter(df, "Age", "Income")
# plot_corr_heatmap(df)
#
# QUICK
# quick_eda(df)
