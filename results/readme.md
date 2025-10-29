# Clustering of Insurance Customers – Results (PCA + KMeans)

> **Project**: Unsupervised customer segmentation  
> **Goal**: Identify risk profiles and optimize pricing  
> **Dataset**: `data_synthetic.csv` 
> **Method**: PCA 2D + KMeans (4 clusters)  
> **Code**: `clustering_pipeline.py` (with `random_state=42`)

---

## Key Findings

| Cluster | Name | Risk | Avg. Profit | Action |
|--------|------|------|-------------|--------|
| **0** | **Gold Safe** | Low | **+3,203 €** | Retain |
| **1** | **High Frequency Loser** | Very High | **+3,114 €** (high cost) | Renegotiate / Exclude |
| **2** | **Balanced Premium** | Medium-High | **+3,643 €** | Maintain |
| **3** | **Claim Heavy, Low Premium** | High | **+1,892 €** | **Increase premium** |

> **Key insight**: **Cluster 3** pays only **1.9k** but has **Claim History = 3.71** → **underpriced by 60%+**

---

## PCA 2D Visualization

![Clustering PCA 2D](clustering_pca_2d.png)

*(Generated with `random_state=42` – fully reproducible)*

- **X-axis (PCA1)**: Premium, Income, Expected Loss  
- **Y-axis (PCA2)**: Claim frequency, Claim History  
- **Cluster 0 (blue)**: bottom-left → low risk, high value  
- **Cluster 1 (red)**: top → high cost  
- **Cluster 3 (green)**: center-left → high claims, low premium

---

## Cluster Summary (mean values)

| Cluster | Sim_Freq | Exp_Loss | Premium | Income | Credit | Prev_Claims | Claim_Hist |
|--------|----------|----------|---------|--------|--------|-------------|------------|
| **0**  | 0.83     | 19.71    | 3223    | 84.7k  | 671    | 1.53        | 1.07       |
| **1**  | 5.91     | 425.28   | 3539    | 81.4k  | 671    | 2.23        | 3.84       |
| **2**  | 2.94     | 147.51   | 3790    | 83.4k  | 665    | 2.10        | 2.62       |
| **3**  | 2.55     | 47.91    | 1939    | 80.4k  | 683    | 1.27        | 3.71       |

---

## Customer Profiles – Description

| Cluster | Profile |
|--------|--------|
| **0 – Gold Safe** | Very few claims, high premium, good credit score. **Ideal customer.** |
| **1 – High Frequency Loser** | 5.9 claims/year, expected loss 425€. **Not sustainable.** |
| **2 – Balanced Premium** | 2.9 claims, pays 3.8k → **best margin.** |
| **3 – Claim Heavy** | Claim History very high (3.71), premium only 1.9k → **underpriced!** |

---

## Recommended Actions

| Cluster | Strategy |
|--------|----------|
| **0** | Loyalty discount, upselling (home, life) |
| **1** | Increase premium to **5.5k+** or **do not renew** |
| **2** | Offer telematics → discount for safe driving |
| **3** | **Increase premium to 3.2k+** (mandatory) |

---

## Technical Pipeline

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features)
    ])),
    ("pca", PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=4, random_state=42))
])
