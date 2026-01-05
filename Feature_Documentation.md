# YUVA Sales Prediction Model - Feature Documentation

## Overview

**Total Features**: 11
**Target Variable**: `tertiary` (Tertiary Sales)

---

## Feature Categories

### 1. Base Features (9 Features)

These are the core features from the dataset that capture business metrics.

| Feature | Description | Why It's Important |
|---------|-------------|-------------------|
| `week` | Week number of the year | Captures weekly seasonality and time progression. Sales often vary by week due to paydays, holidays, and promotional cycles. |
| `ret_stock` | Retail Stock | Indicates inventory levels at retail points. Higher stock availability typically correlates with higher sales potential. |
| `ret_dos` | Retail Days of Stock | Number of days the current retail stock will last. Helps identify stock-out risks and replenishment needs. |
| `dbr_stock` | Distributor Stock | Stock held at distributor level. Reflects supply chain health and product availability. |
| `dbr_dos` | Distributor Days of Stock | Days of stock at distributor level. Indicates supply buffer and potential stockout scenarios. |
| `wod` | Width of Distribution | Number of outlets carrying the product. Wider distribution = more sales touchpoints = higher potential sales. |
| `stocking_outlet` | Number of Stocking Outlets | Outlets that currently have stock. Direct indicator of product availability in market. |
| `activating_outlet` | Number of Activating Outlets | Outlets actively selling the product. Key metric for sales momentum and market penetration. |
| `plc_factor` | Product Life Cycle Factor | Indicates product maturity stage. New products behave differently than mature ones in terms of sales patterns. |

---

### 2. Seasonality Features (2 Features)

These features capture periodic patterns in sales data.

| Feature | Description | Why It's Important |
|---------|-------------|-------------------|
| `seasonality_1` | Primary Seasonality Component | Captures dominant seasonal pattern (e.g., quarterly trends). Helps model understand recurring high/low sales periods. |
| `seasonality_2` | Secondary Seasonality Component | Captures secondary seasonal patterns (e.g., monthly variations). Adds granularity to seasonal modeling. |

**How Seasonality Helps:**
- Mobile phone sales often spike during festivals (Diwali, Eid, Christmas)
- Back-to-school periods affect certain product categories
- Year-end sales and promotional events create predictable patterns

---

## Data Processing Applied

### 1. Log Transformation on Target

```python
y_train_log = np.log1p(y_train)
y_pred = np.expm1(y_pred_log)
```

**Why Log Transform:**
- Reduces skewness in sales data
- Handles large value ranges
- Makes distribution more normal
- Improves model learning

### 2. RobustScaler for Features

```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**Why RobustScaler:**
- Resistant to outliers (uses median and IQR)
- Better than StandardScaler for skewed data
- Preserves relative differences

### 3. Time-Based Split

```python
df_train = 2024 data + 80% of 2025
df_test = last 20% of 2025
```

**Why Time-Based Split:**
- Prevents data leakage
- Simulates real prediction scenario
- More realistic performance estimate

---

## Summary Table: All 11 Features

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | week | Base | Week number |
| 2 | ret_stock | Base | Retail stock |
| 3 | ret_dos | Base | Retail days of stock |
| 4 | dbr_stock | Base | Distributor stock |
| 5 | dbr_dos | Base | Distributor days of stock |
| 6 | wod | Base | Width of distribution |
| 7 | stocking_outlet | Base | Stocking outlets count |
| 8 | activating_outlet | Base | Activating outlets count |
| 9 | plc_factor | Base | Product lifecycle factor |
| 10 | seasonality_1 | Seasonality | Primary seasonal pattern |
| 11 | seasonality_2 | Seasonality | Secondary seasonal pattern |

---

## Model Configuration

### Gradient Boosting Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 300 | Number of boosting stages |
| max_depth | 7 | Maximum tree depth |
| learning_rate | 0.06 | Step size shrinkage |
| subsample | 0.8 | Fraction of samples per tree |
| random_state | 42 | Reproducibility |

---

*Document created for YUVA Sales Prediction Model*
*Last Updated: December 2025*
