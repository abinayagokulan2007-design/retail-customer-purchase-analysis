# Retail Customer Purchase Pattern Analysis

This mini project analyzes retail customer purchase patterns using exploratory data analysis and market basket analysis with the Apriori algorithm.

## Project Structure

- `dataset/`: Optional local dataset storage. If no local dataset is provided, the script downloads the sample dataset automatically.
- `notebooks/`: Jupyter notebooks for additional analysis.
- `outputs/`: Generated outputs including the association rules CSV.
- `report/`: Project reports or summaries.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`

## Setup Instructions

1. Clone or download the project.
2. Create and activate a Python virtual environment in the project folder.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run the analysis script:
   ```bash
   python run_analysis.py
   ```

## Optional Usage

- Load a local dataset file:
  ```bash
  python run_analysis.py --data dataset/Online\ Retail.xlsx
  ```
- Save charts to disk instead of showing them interactively:
  ```bash
  python run_analysis.py --save-charts
  ```
- Change the output directory:
  ```bash
  python run_analysis.py --output-dir outputs
  ```

## Notes

- If `dataset/` is empty, the script will download the sample dataset from the official UCI repository.
- Charts are displayed interactively by default; saved chart images are only produced when `--save-charts` is passed.
- The script writes association rules to `outputs/association_rules.csv`.

## Dataset Columns

The dataset should contain these columns:
- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

## Results

- Top-selling products by quantity.
- Frequent itemsets from market basket analysis.
- Association rules with support, confidence, and lift values.
- Interactive visualizations for top products and rule quality.
