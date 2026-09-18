# Cybersecurity Data Analysis

Exploratory Data Analysis (EDA) project using network traffic data to
identify traffic patterns, attack distributions, statistical
characteristics, outliers, and relationships between network features.

This repository is developed as a practical Data Analyst exercise
using a cybersecurity dataset.

## Project Overview

The analysis focuses on understanding network traffic and identifying
patterns between benign and malicious traffic.

The initial version of this project is intentionally implemented using
standard Pandas, NumPy, and Matplotlib operations without a custom
analysis toolkit.

## Dataset

Dataset: CICIDS2017 sample

The dataset contains network flow information with multiple numerical
network features and a `Label` column representing traffic categories.

Example labels include:

- BENIGN
- DoS
- PortScan
- BruteForce
- WebAttack
- Bot
- Infiltration

The dataset itself is excluded from this repository through
`.gitignore`.

See [`data/README.md`](data/README.md) for dataset information and source.

## Analysis Workflow

The notebook follows a practical EDA workflow:

1. Data loading
2. Dataset inspection
3. Data quality checking
4. Duplicate analysis
5. Missing-value analysis
6. Traffic distribution analysis
7. Attack distribution analysis
8. Statistical analysis
9. Outlier analysis
10. Correlation analysis
11. Data visualization
12. Key findings

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook
