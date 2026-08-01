# SmartCart — Customer Segmentation

Unsupervised learning project that segments retail customers into distinct groups based on demographics, income, and spending behavior, using PCA for dimensionality reduction and clustering (KMeans + Agglomerative Clustering) to identify customer personas.

## 📊 Dataset

`smartcart.csv` — 2,240 customers with the following fields:

| Category | Fields |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Loyalty | `Dt_Customer` (enrollment date), `Recency` (days since last purchase) |
| Spending | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase behavior | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Engagement | `Complain`, `Response` |


## 🧠 Pipeline

1. **Data preprocessing** — impute missing `Income` values (median), engineer new features:
   - `Age` (from `Year_Birth`)
   - `Customer_Tenure_Days` (from `Dt_Customer`)
   - `Total_spending` (sum of all `Mnt*` columns)
   - `Total_children` (`Kidhome` + `Teenhome`)
   - Simplified `Education` (Undergraduate / Graduate / Postgraduate) and `Living_with` (Alone / Partner) categories
2. **Outlier removal** — drop customers with `Age` ≥ 90 or `Income` ≥ 600,000
3. **EDA** — correlation heatmap, pairplots of key numeric features
4. **Encoding & scaling** — one-hot encode categoricals, standardize all features
5. **Dimensionality reduction** — PCA to 3 components for visualization and clustering input
6. **Optimal cluster selection** — elbow method (WCSS) + silhouette score
7. **Clustering** — KMeans and Agglomerative Clustering, both with k = 4
8. **Cluster profiling** — visualize income vs. spending by cluster, summarize each cluster's average feature values

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

## 🚀 Usage

```bash
jupyter notebook smartcart.ipynb
```

Run all cells top to bottom. Make sure `smartcart.csv` is in the same directory as the notebook.

## 📁 Repository Structure

```
.
├── smartcart.ipynb     # Main analysis notebook
├── smartcart.csv        # Raw customer dataset
└── README.md
```

## 📈 Output

The notebook produces four customer segments differentiated primarily by income and total spending, along with visual cluster summaries (3D PCA scatter plots, cluster size counts, and income-vs-spending scatter by cluster) that can inform targeted marketing strategies.

## 📄 License

Add a license of your choice (e.g., MIT) if you plan to share this publicly.
