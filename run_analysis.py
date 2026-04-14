#!/usr/bin/env python
# coding: utf-8

"""Retail customer purchase pattern analysis.

This script supports interactive chart display and optional chart saving.
It can load a local dataset file or download the sample dataset if no local file is found.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

REMOTE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CHART_DIR = DEFAULT_OUTPUT_DIR / "charts"
DEFAULT_LOCAL_DATASET_PATHS = [
    Path("dataset/Online Retail.xlsx"),
    Path("dataset/Online Retail.csv"),
    Path("dataset/online_retail.csv"),
]

sns.set(style="whitegrid")
plt.style.use("seaborn-v0_8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze retail customer purchase patterns with market basket analysis."
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        help="Path to a local retail dataset file (.xlsx or .csv)."
    )
    parser.add_argument(
        "--save-charts",
        action="store_true",
        help="Save chart images to outputs/charts/. By default charts are displayed interactively.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.01,
        help="Minimum support for Apriori frequent itemset mining.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for association rules.",
    )
    return parser.parse_args()


def create_output_directories(output_dir: Path, save_charts: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_charts:
        (output_dir / "charts").mkdir(parents=True, exist_ok=True)


def locate_local_dataset(local_path: Path | None) -> Path | None:
    if local_path is not None:
        if local_path.exists():
            return local_path
        raise FileNotFoundError(f"Dataset not found at: {local_path}")

    for candidate in DEFAULT_LOCAL_DATASET_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_dataset(path: Path | None) -> pd.DataFrame:
    dataset_path = locate_local_dataset(path)
    if dataset_path is not None:
        print(f"Loading local dataset from: {dataset_path}")
        suffix = dataset_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(dataset_path)
        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        raise ValueError("Unsupported dataset extension: {dataset_path.suffix}")

    print("No local dataset found. Downloading remote dataset...")
    return pd.read_excel(REMOTE_URL)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    original_shape = data.shape
    data = data.dropna(subset=["CustomerID", "Description"])
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    data["Description"] = data["Description"].astype(str).str.strip().str.upper()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    print(f"Cleaned data: {original_shape} -> {data.shape}")
    return data


def show_or_save_chart(save: bool, path: Path | None = None) -> None:
    if save and path is not None:
        plt.tight_layout()
        plt.savefig(path)
        print(f"Saved chart: {path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_top_products(top_products: pd.Series, save: bool, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    top_products.plot(kind="bar")
    plt.title("Top 10 Products by Quantity Sold")
    plt.xlabel("Product Description")
    plt.ylabel("Total Quantity")
    plt.xticks(rotation=45, ha="right")
    chart_path = output_dir / "charts" / "top_products.png"
    show_or_save_chart(save, chart_path if save else None)


def plot_support_confidence(rules: pd.DataFrame, save: bool, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(rules["support"], rules["confidence"], alpha=0.5)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support vs Confidence")
    chart_path = output_dir / "charts" / "support_confidence_scatter.png"
    show_or_save_chart(save, chart_path if save else None)


def plot_top_rules(top_rules: pd.DataFrame, save: bool, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_rules)), top_rules["lift"])
    plt.xticks(
        range(len(top_rules)),
        [f"{list(a)} -> {list(c)}" for a, c in zip(top_rules["antecedents"], top_rules["consequents"])],
        rotation=90,
    )
    plt.title("Top 10 Association Rules by Lift")
    plt.ylabel("Lift")
    chart_path = output_dir / "charts" / "rules_lift_bar.png"
    show_or_save_chart(save, chart_path if save else None)


def run_analysis(args: argparse.Namespace) -> None:
    create_output_directories(args.output_dir, args.save_charts)
    df = load_dataset(args.data)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}\n")

    print("[2] Cleaning data...")
    df = clean_data(df)
    print(df.head())

    print("[3] Exploratory Data Analysis...")
    print(f"Total transactions: {df['InvoiceNo'].nunique()}")
    print(f"Unique products: {df['StockCode'].nunique()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")

    top_products = (
        df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    )
    print("\nTop 10 products by quantity:")
    print(top_products)
    plot_top_products(top_products, args.save_charts, args.output_dir)

    print("\n[4] Frequency tables...")
    product_freq = df["Description"].value_counts().head(20)
    print(product_freq)
    country_freq = df["Country"].value_counts().head(10)
    print(country_freq)

    print("\n[5] Preparing data for market basket analysis...")
    basket = df.groupby("InvoiceNo")["Description"].apply(list).reset_index()
    transactions = basket["Description"].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"Basket shape: {basket_df.shape}")

    print("\n[6] Finding frequent itemsets with Apriori...")
    frequent_itemsets = apriori(
        basket_df, min_support=args.min_support, use_colnames=True
    ).sort_values(by="support", ascending=False)
    print(frequent_itemsets.head(10))

    print("\n[7] Generating association rules...")
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=args.min_confidence
    ).sort_values(by="lift", ascending=False)
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

    print("\n[8] Creating visualizations...")
    plot_support_confidence(rules, args.save_charts, args.output_dir)
    plot_top_rules(rules.head(10), args.save_charts, args.output_dir)

    output_path = args.output_dir / "association_rules.csv"
    rules.to_csv(output_path, index=False)
    print(f"Association rules saved to {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nOutputs generated:")
    print(f"  - {output_path}")
    if args.save_charts:
        print(f"  - {args.output_dir / 'charts'}")
    else:
        print("Charts displayed interactively.")


if __name__ == "__main__":
    run_analysis(parse_args())
