# Titanic Training: End-to-End Data Pipeline

[![Python 3.10+](https://shields.io)](https://python.org)
[![Pandas](https://shields.io)](https://pydata.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)

An engineering-focused workspace demonstrating optimized feature pipelines, robust data cleaning, targeted imputation loops, and predictive classification modeling on the Titanic passenger dataset.

---

##  Features

*   **Algorithmic Imputation:** Maps missing passenger details using name-based lookup maps instead of generic statistical means.
*   **Targeted Record Correction:** Implements precision conditional masks to manually clean verified historical anomalies.
*   **Hybrid Feature Encoding Engine:** Combining custom `OneHotEncoder` structures for descriptive fields alongside `LabelEncoder` formatting for complex strings.
*   **Leakage Prevention Setup:** Strict isolation of features (X) and target outcomes (y) prior to array conversion.

---

##  Repository Layout

```text
├── titanic.csv               # Baseline target raw dataset
├── Titanic_Ages.csv          # Targeted variant containing structural updates
├── Feature_Engineering.ipynb # Interactive workspace documenting cleaning & modeling
└── README.md                 # Project architecture documentation
```

---

## 🛠️ Step-by-Step Architecture

### 1. Diagnostics & File Auditing
Leverages host environment tools (`os.getcwd()`) to construct rigid relative path strings and validates dataset properties using memory footprint summaries (`.info()`).

### 2. Imputation & Boundary Patching
Deploys duplicate filtering indices to map missing details across identical passenger identifiers. Applies string matching logic to isolate and correct standalone missing values.

### 3. Feature Matrix Splitting
Isolates predictor variables from outcome values using error-safeguarded exclusions (`errors='ignore'`) to prevent execution failures during pipeline recalculations.

### 4. Categorical Transformation
Iterates column data streams dynamically. Converts localized string values into structural binary matrices suitable for regression algorithms:

```python
for c in x.columns:
    if c in columns_to_encode:
        one_hot = pd.get_dummies(x[c], prefix=c, dtype=int)
        train1 = pd.concat([train1, one_hot], axis=1)
    elif x[c].dtype == 'object':
        train1[c] = label.fit_transform(x[c].astype(str))
    else:
        train1[c] = x[c]
```

---

##  Quick Start

### Installation
Deploy core numeric frameworks into your local virtual environment:
```bash
pip install pandas numpy scikit-learn notebook
```

### Run the Pipeline
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com
   cd Titanic_Training
   ```
2. Launch the workspace environment:
   ```bash
   jupyter notebook Feature_Engineering.ipynb
   ```
3. Execute sequentially via **Kernel** $\rightarrow$ **Restart & Run All**.
