# Retail Customer Purchase Pattern Analysis

This mini project analyzes retail customer purchase patterns using data analytics techniques, including exploratory data analysis (EDA) and market basket analysis with the Apriori algorithm.

## Project Structure

- `dataset/`: Contains the retail transaction dataset (CSV format).
- `notebooks/`: Jupyter notebooks for the analysis.
- `outputs/`: Generated outputs including charts and association rules.
  - `charts/`: Saved visualization charts.
- `report/`: Project reports or summaries.

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`

## Setup Instructions

1. Clone or download the project.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the retail transaction dataset (CSV) in the `dataset/` folder. The dataset should include columns like InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
4. Open the Jupyter notebook in `notebooks/retail_analysis.ipynb`.
5. Run the cells to perform the analysis.

## Analysis Overview

1. **Data Loading and Cleaning**: Load CSV, remove nulls, filter valid transactions, standardize data.
2. **Exploratory Data Analysis**: Summary statistics, top products, visualizations.
3. **Market Basket Analysis**: Use Apriori to find frequent itemsets and association rules.
4. **Visualizations**: Bar charts, heatmaps, scatter plots.
5. **Outputs**: Save charts and rules to `outputs/`.

## Dataset

The project uses a sample online retail dataset. Ensure the CSV has the following columns:
- InvoiceNo: Invoice number
- StockCode: Product code
- Description: Product description
- Quantity: Quantity purchased
- InvoiceDate: Date of transaction
- UnitPrice: Price per unit
- CustomerID: Customer identifier
- Country: Country of customer

## Results

- Top-selling products identified.
- Association rules with support, confidence, lift metrics.
- Visual insights into purchase patterns.