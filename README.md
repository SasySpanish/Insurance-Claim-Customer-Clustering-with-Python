# Insurance Customer Risk Segmentation & Pricing Optimization

> **Goal**: Use synthetic insurance data to **simulate claims**, **discover risk profiles**, and **optimize pricing** via unsupervised machine learning.  
> **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, PCA + KMeans  
> **Key Outcome**: 4 actionable customer clusters with **pricing recommendations**  
> **License**: MIT

---

## Project Overview

This project demonstrates a **full data science pipeline** for insurance risk analytics:

1. **Raw synthetic customer data** → `data/`
2. **Monte Carlo claim simulation + clustering** → `src/`
3. **Visualizations + actionable insights** → `results/`

The result: **4 customer risk segments** that reveal **underpriced policies**, **high-risk losers**, and **profit drivers**.

---

## Project Structure

```
insurance-risk-segmentation/
│
├── data/                  # Raw input data
│   ├── data_synthetic.csv
│   └── README.md          # Dataset description
│
├── src/                   # Analysis code
│   ├── analysis_pipeline.py
│   └── README.md          # Step-by-step pipeline explanation
│
├── results/               # Outputs & insights
│   ├── cluster_summary.csv
│   ├── clustering_pca_2d.png
│   ├── frequency_histogram.png
│   ├── expected_loss_histogram.png
│   ├── severity_histogram.png
│   ├── freq_vs_sev_scatter.png
│   └── README.md          # Full results & business actions
│
└── README.md              # This file
```

---

## Key Results

| Cluster | Name | Risk | Avg. Premium | Expected Loss | Profit | Action |
|--------|------|------|--------------|----------------|--------|--------|
| **0** | **Gold Safe** | Low | 3,223 € | 19.71 € | **+3,203 €** | Retain & upsell |
| **1** | **High Frequency Loser** | Very High | 3,539 € | 425 € | **+3,114 €** | Increase premium or exclude |
| **2** | **Balanced Premium** | Medium-High | 3,790 € | 147 € | **+3,643 €** | Maintain & reward |
| **3** | **Claim Heavy, Low Premium** | High | **1,939 €** | 47.91 € | **+1,892 €** | **Increase to 3.2k+** |

> **Critical Insight**:  
> **Cluster 3** has **Claim History = 3.71** (near max) but pays **only 1.9k** → **underpriced by ~60%**.

---

## Visual Highlights

| Plot | Insight |
|------|--------|
| **PCA 2D Clustering** | 4 clear, compact groups → interpretable risk segments |
| **Claim Frequency** | Right-skewed: most have 0–1 claims |
| **Expected Loss** | Long tail: few high-cost customers dominate risk |
| **Frequency vs Severity** | Strong correlation; `Risk Profile` drives upper tail |

See all plots in [`results/`](results/)

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourname/insurance-risk-segmentation.git
cd insurance-risk-segmentation

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run the analysis
python src/analysis_pipeline.py
```

> Outputs saved automatically to `results/`

---

## Business Impact

| Action | Estimated Benefit |
|-------|-------------------|
| Reprice **Cluster 3** (+1.3k/customer) | **+65 M€** (50k customers) |
| Exclude or penalize **Cluster 1** | Reduce loss exposure |
| Upsell **Cluster 0 & 2** | Increase retention & revenue |

---

## Reproducibility

- **Deterministic**: `random_state=42` in PCA, KMeans, and simulation
- **Pipeline-based**: Consistent preprocessing
- **Documented**: Full READMEs in every folder

---

## Author

**[Salvatore Spagnuolo]**  

