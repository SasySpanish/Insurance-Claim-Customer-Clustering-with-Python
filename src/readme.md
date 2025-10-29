# Insurance Customer Risk Analysis & Clustering Pipeline

> **File**: `analysis_pipeline.py`  
> **Purpose**: Complete end-to-end analysis of synthetic insurance customer data:  
> 1. **Monte Carlo simulation of claims**  
> 2. **Exploratory Data Analysis (EDA)**  
> 3. **Unsupervised customer segmentation via PCA + KMeans**  
> 4. **Risk profiling and business insights**  
>  
> **Input**: `data/data_synthetic.csv`  
> **Outputs**:  
> - Interactive EDA plots  
> - `results/cluster_summary.csv`  
> - 2D PCA clustering visualization  

---

## Workflow Overview

This script transforms raw customer data into **actionable risk segments** using a robust, reproducible machine learning pipeline. It simulates realistic claim behavior, explores key risk patterns, and groups customers into **four distinct risk profiles**.

---

## 1. Monte Carlo Claim Simulation

The pipeline simulates **realistic claim frequency and severity** using statistical distributions:

### Claim Frequency
- Modeled with a **Poisson distribution**.
- Base rate adjusted by:
  - Past claim history
  - Previous claims
  - Risk profile (higher risk → higher expected claims)
- Result: `Sim_Frequency` — number of claims per customer.

### Claim Severity
- Modeled with a **log-normal distribution** (heavy-tailed, realistic payouts).
- Mean scaled with premium and uplifted by:
  - Policy type (business > family > group)
  - Risk profile
  - Claim history
- Each claim is simulated individually.

### Derived Risk Metrics
- `Sim_Total_Loss`: Total payout across all claims.
- `Expected_Loss`: Smoothed estimate of average annual loss per customer.

---

## 2. Exploratory Data Analysis (EDA)

Four diagnostic plots reveal the **true risk distribution**:

| Plot | Insight |
|------|--------|
| **Claim Frequency Histogram** | Highly right-skewed: most customers have 0–1 claims; rare high-frequency cases. |
| **Total Severity Histogram** | Long tail: most policies low-cost, few catastrophic losses. |
| **Expected Loss Histogram** | Confirms risk concentration: majority near zero, tail up to thousands. |
| **Frequency vs Severity Scatter** | Strong positive correlation; higher `Risk Profile` = upper-right quadrant. |

These visualizations **validate the simulation** and highlight natural customer groupings.

---

## 3. Customer Segmentation (Clustering)

### Feature Selection
12 key variables are used:
- Demographics: `Age`, `Income Level`
- Policy: `Premium Amount`, `Policy Type`
- Risk indicators: `Risk Profile`, `Credit Score`, `Driving Record`
- Simulated: `Sim_Frequency`, `Expected_Loss`
- History: `Claim History`, `Previous Claims History`

### Preprocessing
- **Numerical features**: Standardized (zero mean, unit variance).
- **Categorical features**: One-hot encoded (no dummy variable trap).

### Dimensionality Reduction
- **PCA** reduces data to 2D for visualization and noise reduction.
- Components are **reproducible** (`random_state=42`).

### Clustering
- **KMeans** groups customers into **4 clusters**.
- Fully **reproducible** (`random_state=42`).

---

## 4. Cluster Interpretation & Business Insights

The resulting clusters represent **four clear customer risk profiles**:

| Cluster | Profile | Key Traits | Business Action |
|--------|--------|-----------|-----------------|
| **0** | **Gold Safe** | Low frequency, near-zero loss, high premium | Retain & upsell |
| **1** | **High Frequency Loser** | 5.9 claims/year, high expected loss | Increase premium or exclude |
| **2** | **Balanced Premium** | Moderate claims, high premium → best margin | Maintain & reward |
| **3** | **Claim Heavy, Low Premium** | High claim history, but underpriced | **Urgent repricing** |

> **Critical Insight**: Cluster 3 has **high claim history but pays only ~1.9k** — **underpriced by 60%+**.

---

## 5. Output & Reporting

- **Cluster Summary**: Mean values per group exported to `results/cluster_summary.csv`.
- **PCA 2D Plot**: Visual separation of risk profiles.
- **All results** stored in `/results/` for reporting and integration.

---

## Reproducibility & Robustness

- **Deterministic simulation** via fixed random state.
- **Stable PCA and KMeans** using `random_state=42`.
- **Pipeline design** ensures consistent preprocessing and modeling.

---

## Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## How to Run

```bash
python analysis_pipeline.py
