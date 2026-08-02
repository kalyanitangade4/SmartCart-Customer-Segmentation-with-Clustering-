# SmartCart — Customer Segmentation Using Unsupervised Learning

An end-to-end data science project that segments retail customers into distinct behavioral personas using demographic, income, and purchasing data. The pipeline combines feature engineering, PCA-based dimensionality reduction, and clustering (KMeans and Agglomerative Clustering) to uncover actionable customer groups for targeted marketing.

---

## 🔍 Project Overview

SmartCart analyzes ~2,240 retail customers and groups them into four clusters based on shared characteristics such as income level, total spend, product category preferences, and purchase channel behavior (web, catalog, in-store). The goal is to move from a one-size-fits-all marketing approach to data-driven, segment-specific strategies.

The notebook walks through the full unsupervised learning workflow:

`Raw data → Cleaning → Feature Engineering → Outlier Removal → EDA → Encoding & Scaling → PCA → Optimal K Selection → Clustering → Cluster Profiling`

---

## 💼 Business Problem

Retailers often struggle to tailor marketing spend efficiently when treating all customers the same way. By segmenting customers based on income, spending habits, and engagement, a business can:

- Identify high-value customers worth retention campaigns
- Spot price-sensitive segments that respond to deals/discounts
- Understand which channels (web/store/catalog) each segment prefers
- Personalize promotions instead of broad, low-conversion campaigns

---

## 📊 Dataset

**File:** `smartcart.csv` (2,240 rows × 22 columns)

| Category | Columns | Description |
|---|---|---|
| **Identity** | `ID` | Unique customer identifier |
| **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income` | Basic customer profile |
| **Household** | `Kidhome`, `Teenhome` | Number of children/teens at home |
| **Loyalty** | `Dt_Customer`, `Recency` | Enrollment date, days since last purchase |
| **Spend by category** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Amount spent (last 2 years) per category |
| **Purchase channel** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` | Purchase counts by channel |
| **Engagement** | `Complain`, `Response` | Complaint filed, responded to last campaign |

---

## 🛠️ Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

**Python packages:**

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization (heatmaps, pairplots) |
| `scikit-learn` | Preprocessing, PCA, clustering, metrics |
| `kneed` | Automatic elbow-point detection for optimal K |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/smartcart.git
cd smartcart

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

```bash
pip install -r requirements.txt
```

---

## 🧠 Methodology

### 1. Data Preprocessing
- Load `smartcart.csv` into a pandas DataFrame
- Inspect missing values (`df.isnull().sum()`)
- Impute missing `Income` values with the **median** (robust to outliers/skew)

### 2. Feature Engineering
| New Feature | Derived From | Logic |
|---|---|---|
| `Age` | `Year_Birth` | `2026 - Year_Birth` |
| `Customer_Tenure_Days` | `Dt_Customer` | Days between enrollment date and the most recent enrollment in the dataset |
| `Total_spending` | All `Mnt*` columns | Sum of spend across all product categories |
| `Total_children` | `Kidhome` + `Teenhome` | Combined household dependents |
| `Education` (simplified) | `Education` | Grouped into `Undergraduate`, `Graduate`, `Postgraduate` |
| `Living_with` | `Marital_Status` | Grouped into `Alone` or `Partner` |

### 3. Column Cleanup
Dropped raw/redundant columns once engineered equivalents exist: `ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the individual `Mnt*` spend columns (superseded by `Total_spending`).

### 4. Outlier Removal
- Removed customers with `Age` ≥ 90 (likely data entry errors)
- Removed customers with `Income` ≥ 600,000 (extreme outliers skewing scale)
- Verified visually using `seaborn.pairplot` before and after filtering

### 5. Exploratory Data Analysis
- Correlation heatmap (`seaborn.heatmap`) to check multicollinearity across numeric features
- Pairplots to visualize relationships between income, recency, response, age, spending, and children

### 6. Feature Encoding & Scaling
- One-hot encoded categorical columns (`Education`, `Living_with`) using `OneHotEncoder`
- Standardized all features using `StandardScaler` (zero mean, unit variance) — required since clustering algorithms are distance-based and sensitive to feature scale

### 7. Dimensionality Reduction (PCA)
- Reduced the scaled feature set to **3 principal components**
- Visualized customers in 2D and 3D PCA space
- Checked `explained_variance_ratio_` to confirm the components capture meaningful variance

### 8. Determining Optimal Cluster Count (K)
Two complementary methods were used:
- **Elbow Method**: plotted Within-Cluster Sum of Squares (WCSS) across K = 1–10, with the elbow point auto-detected using the `kneed` library
- **Silhouette Score**: measured cluster cohesion/separation across K = 2–10
- Both methods were plotted together on a dual-axis chart for comparison

### 9. Clustering
Two algorithms were applied with **K = 4**, chosen from the elbow/silhouette analysis:
- **KMeans** — centroid-based, fast, works well on PCA-reduced continuous space
- **Agglomerative Clustering** (Ward linkage) — hierarchical, validates KMeans results from a different algorithmic approach

Both were visualized as 3D scatter plots colored by cluster label.

### 10. Cluster Characterization
- Count of customers per cluster (`sns.countplot`)
- Income vs. Total Spending scatter plot, colored by cluster — the primary lens for interpreting segments
- Groupby summary (`X.groupby("labels").mean()`) showing average feature values per cluster, used to name/interpret each persona (e.g., high-income/high-spend "premium" customers vs. budget-conscious "value" customers)

---

## 📈 Results

The final model identifies **4 customer segments**, primarily differentiated along the income–spending axis, with secondary differences in household size, education, and preferred purchase channel. Example segment framing (based on typical output from this pipeline — refer to the notebook's `cluster_summary` output for exact values on your data):

| Cluster | Typical Profile |
|---|---|
| 0 | Lower income, lower spend — price-sensitive, deal-driven |
| 1 | Mid income, moderate spend — balanced, web + store shoppers |
| 2 | High income, high spend — premium customers, catalog/store heavy |
| 3 | Mixed household size, moderate income — family-oriented buyers |

> Exact cluster definitions depend on the data run — check the `cluster_summary` DataFrame printed at the end of the notebook for precise averages.

---

## ▶️ Usage

```bash
jupyter notebook smartcart.ipynb
```

Run all cells sequentially (`Kernel → Restart & Run All`). Ensure `smartcart.csv` sits in the same directory as the notebook, or update the file path in the `pd.read_csv(...)` call at the top of the notebook.

---

## 🔮 Future Improvements

- Add a `requirements.txt` or `environment.yml` for full reproducibility
- Try alternative clustering algorithms (DBSCAN, Gaussian Mixture Models) for comparison
- Add a train/holdout split to validate cluster stability over time
- Build a simple dashboard (e.g., Streamlit) to explore clusters interactively
- Translate cluster personas into concrete marketing recommendations per segment

---

## 👨‍💻 Author
Kalyani Tangade | B.Tech IT Graduate | Aspiring Data Scientist 📍 Mumbai, India
