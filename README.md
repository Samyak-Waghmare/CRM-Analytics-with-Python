# Customer Analytics: RFM Segmentation & CLTV Prediction

End-to-end customer analytics pipeline built on 19,945 omnichannel retail records from FLO (a Turkish shoe retailer). The project covers behavioral segmentation using RFM analysis and probabilistic Customer Lifetime Value (CLTV) prediction using BG/NBD and Gamma-Gamma models.

---

## Project Structure

```
.
├── RFM_FLO_project.py        # RFM segmentation pipeline
├── FLO_CLTV_Prediction.py    # CLTV prediction pipeline
├── dataset/
│   └── flo_data_20k.csv      # Raw dataset (19,945 customers)
├── target_customers_new_brand.csv   # Output: high-value women category customers
└── target_customers_discount.csv    # Output: win-back targets for discount campaign
```

---

## Part 1 — RFM Segmentation

Customers are scored on 3 behavioral dimensions:

| Metric | Definition |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Total number of purchases |
| **Monetary** | Total amount spent |

Each metric is binned into quintile scores (1–5) and customers are mapped to **10 named segments**:

| Segment | Description |
|---|---|
| Champions | Bought recently, buy often, spend the most |
| Loyal Customers | Regular buyers with high frequency |
| Potential Loyalists | Recent customers with growing frequency |
| At Risk | Previously good customers going inactive |
| Cant Loose | High-value customers who have stopped buying |
| Hibernating | Low recency, low frequency, low spend |
| New Customers | Bought very recently for the first time |
| ... | etc. |

### Key Results

- **1,932 Champions** — avg. recency 17 days, avg. spend 1,406
- **3,131 At-Risk customers** — avg. recency 241 days (inactive 8+ months)
- **2,497 customers** identified for a premium women's brand launch campaign
- Separate customer list exported for a 40% Men/Children discount campaign

---

## Part 2 — CLTV Prediction

Predicts the **6-month future revenue** each customer is expected to generate using two probabilistic models:

| Model | Purpose |
|---|---|
| **BG/NBD** (Beta-Geometric / Negative Binomial Distribution) | Predicts expected number of future purchases |
| **Gamma-Gamma** | Predicts expected average transaction value |

CLTV = Expected Purchases × Expected Average Value (discounted over 6 months)

### Key Results

| Segment | Avg. 6-Month CLTV |
|---|---|
| A (Top 25%) | 372.51 |
| B | 200.41 |
| C | 138.49 |
| D (Bottom 25%) | 80.01 |

**Segment A customers generate 4.7× more projected revenue than Segment D.**

---

## Tech Stack

- Python 3.x
- pandas
- [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) — BG/NBD and Gamma-Gamma models
- datetime

## Installation

```bash
pip install pandas lifetimes
```

## Usage

```bash
# Run RFM segmentation
python RFM_FLO_project.py

# Run CLTV prediction
python FLO_CLTV_Prediction.py
```
