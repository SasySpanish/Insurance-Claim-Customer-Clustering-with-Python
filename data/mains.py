import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
df = pd.read_csv("data_synthetic.csv")

# ------------------------------------------------------------
def simulate_frequency(row):
    lam = 0.4 + row["Claim History"]*0.3 + row["Previous Claims History"]*0.2
    # Risk Profile adjustments
    if row["Risk Profile"] == 1: lam *= 1.3
    elif row["Risk Profile"] == 2: lam *= 1.8
    elif row["Risk Profile"] == 3: lam *= 2.2
    return np.random.poisson(lam)

df["Sim_Frequency"] = df.apply(simulate_frequency, axis=1)

def simulate_severity(row):
    mu = np.log(row["Premium Amount"]) - 5
    # Policy Type adjustments
    if row["Policy Type"] == "group": mu += 0.3
    elif row["Policy Type"] == "family": mu += 0.4
    elif row["Policy Type"] == "business": mu += 0.5
    # Risk Profile adjustments
    if row["Risk Profile"] == 2: mu += 0.3
    elif row["Risk Profile"] == 3: mu += 0.5
    # Previous claims adjustment
    mu += row["Previous Claims History"] * 0.2
    sigma = 0.8
    return np.random.lognormal(mean=mu, sigma=sigma)

sim_losses = []
for i in range(len(df)):
    freq = df.loc[i, "Sim_Frequency"]
    losses = [simulate_severity(df.loc[i]) for _ in range(freq)]
    sim_losses.append(np.sum(losses))

df["Sim_Total_Loss"] = sim_losses
df["Expected_Loss"] = df["Sim_Frequency"] * df["Sim_Total_Loss"] / (df["Sim_Frequency"] + 1)
# 1. Distribuzione frequenza sinistri
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Sim_Frequency"], bins=range(0, df["Sim_Frequency"].max()+2), kde=False, color='skyblue')
plt.title("Distribuzione della frequenza dei sinistri (Sim_Frequency)")
plt.xlabel("Numero di sinistri")
plt.ylabel("Numero di clienti")
plt.show()

# ------------------------------------------------------------
# 2. Distribuzione severità totale
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Sim_Total_Loss"], bins=50, kde=True, color='salmon')
plt.title("Distribuzione della severità totale dei sinistri (Sim_Severity_Total)")
plt.xlabel("Importo totale sinistri")
plt.ylabel("Numero di clienti")
plt.show()

# ------------------------------------------------------------
# 3. Distribuzione perdita attesa
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Expected_Loss"], bins=50, kde=True, color='red')
plt.title("Distribuzione della perdita attesa (Expected_Loss)")
plt.xlabel("Perdita attesa")
plt.ylabel("Numero di clienti")
plt.show()

# ------------------------------------------------------------
# 4. Scatter Frequency vs Severity
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="Expected_Loss", y="Sim_Total_Loss", hue="Risk Profile", data=df, alpha=0.7)
plt.title("Frequency vs Severity per cliente")
plt.xlabel("Sim_Frequency")
plt.ylabel("Sim_Total_Loss")
plt.show()

# 5. Clustering
features = [
    "Age", "Gender", "Income Level", "Premium Amount",
    "Risk Profile", "Credit Score", "Driving Record",
    "Sim_Frequency", "Expected_Loss", "Previous Claims History",
    "Claim History", "Policy Type"
]

X = df[features]

# -----------------------------
# 2. Separazione numeriche/categoriali
# -----------------------------
num_features = [
    "Age", "Income Level", "Premium Amount", "Credit Score",
    "Sim_Frequency", "Expected_Loss", "Previous Claims History", "Claim History", "Risk Profile"
]

cat_features = [
    "Gender", "Policy Type", "Driving Record"
]

# -----------------------------
# 3. Preprocessing + Clustering
# -----------------------------
preproc = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

pipeline = Pipeline([
    ("prep", preproc),
    ("pca", PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=4, random_state=42))
])

# Fit e predizione cluster
df["Cluster"] = pipeline.fit_predict(X)

# -----------------------------
# 4. Visualizzazione PCA
# -----------------------------
X_pca = pipeline["pca"].transform(preproc.fit_transform(X))

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["Cluster"], palette="tab10")
plt.title("Clustering clienti - PCA 2D")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


# -----------------------------
# 5. Sintesi cluster
# -----------------------------
cluster_summary = df.groupby("Cluster")[
    ["Sim_Frequency","Expected_Loss","Premium Amount","Income Level","Credit Score",
     "Previous Claims History","Claim History"]
].mean().round(2)

cluster_summary.to_csv('cluster_summary.csv', index=False)

print("\nCluster summary (valori medi):")
print(cluster_summary)