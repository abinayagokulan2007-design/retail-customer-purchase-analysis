#!/usr/bin/env python
# coding: utf-8

# Retail Customer Purchase Pattern Analysis

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# Set style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8')

print("="*80)
print("RETAIL CUSTOMER PURCHASE PATTERN ANALYSIS")
print("="*80)

# Load the dataset
print("\n[1] Loading Dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df = pd.read_excel(url)
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}\n")

# Data Cleaning
print("[2] Data Cleaning and Preprocessing...")
print("Original shape:", df.shape)

# Remove rows with null CustomerID or Description
df = df.dropna(subset=['CustomerID', 'Description'])

# Remove cancelled transactions (InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove transactions with negative quantity or zero price
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Standardize product descriptions (strip whitespace, uppercase)
df['Description'] = df['Description'].str.strip().str.upper()

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Add a TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("After cleaning shape:", df.shape)
print("Cleaned data sample:")
print(df.head())

# Exploratory Data Analysis
print("\n[3] Exploratory Data Analysis...")
print(f"Total transactions: {df['InvoiceNo'].nunique()}")
print(f"Unique products: {df['StockCode'].nunique()}")
print(f"Unique customers: {df['CustomerID'].nunique()}")

# Top-selling products by quantity
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 products by quantity:")
print(top_products)

# Save top 10 products
print("\nDisplaying top products chart...")
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Frequency tables
print("\n[4] Generating Frequency Tables...")
product_freq = df['Description'].value_counts().head(20)
print("Top 20 product frequencies:")
print(product_freq)

country_freq = df['Country'].value_counts().head(10)
print("\nTop 10 countries by transactions:")
print(country_freq)

# Market Basket Analysis
print("\n[5] Preparing Data for Market Basket Analysis...")
basket = df.groupby('InvoiceNo')['Description'].apply(list).reset_index()
transactions = basket['Description'].tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Basket shape: {basket_df.shape}")

print("\n[6] Finding Frequent Itemsets with Apriori...")
frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print("Frequent itemsets (min support 0.01):")
print(frequent_itemsets.head(10))

print("\n[7] Generating Association Rules...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(by='lift', ascending=False)

print("Association rules (min confidence 0.5):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Visualizations
print("\n[8] Creating Visualizations...")
plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()
print("Displayed: support_confidence_scatter")

top_rules = rules.head(10)
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_rules)), top_rules['lift'])
plt.xticks(range(len(top_rules)), [f"{list(a)} -> {list(c)}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])], rotation=90)
plt.title('Top 10 Association Rules by Lift')
plt.ylabel('Lift')
plt.tight_layout()
plt.show()
print("Displayed: rules_lift_bar")

# Save outputs
print("\n[9] Saving Outputs...")
rules.to_csv('outputs/association_rules.csv', index=False)
print("Association rules saved to outputs/association_rules.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nOutputs generated:")
print("  - outputs/association_rules.csv")
print("\nCharts displayed interactively:")
