# Online Shopping Purchase Prediction

## Project Overview
This project focuses on predicting whether a user will complete a purchase during an online shopping session. The problem is framed as a **binary classification task**, where the target variable is:

- **PurchaseCompleted** (0 = No Purchase, 1 = Purchase)

The workflow includes:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation using multiple machine learning algorithms

---

## Dataset Notes
Due to dataset usage restrictions, the original dataset is not included in this repository.

- A **sample dataset** is provided for demonstration purposes.
- The **results shown below were obtained using the full dataset** during internal testing.

---

## Exploratory Data Analysis (EDA)

### Key Insights
- The dataset is **imbalanced**, with a purchase rate of approximately **15.4%**
- **PageValue** and **ProductPageTime** are highly right-skewed
- **BounceRate** and **ExitRate** show strong correlation (~0.86)
- Certain categories in:
  - TrafficSource
  - GeographicRegion
  - CustomerType  
  show differing purchase behaviors

---

## Data Preprocessing

- Negative values replaced with NaN:
  - `BounceRate`, `ProductPageTime`

- Missing value handling:
  - Numerical → Median imputation
  - Categorical → Labeled as "Unknown"

- Categorical cleaning:
  - Rare `TrafficSource` grouped into **"Others"**
  - Missing `TrafficSource` retained as **"Unknown"** to capture missingness signal
  - `CustomerType` simplified into:
    - New Visitor
    - Returning Visitor
    - Unregistered

---

## Feature Engineering

- Log transformation applied:
  - `PageValue`
  - `ProductPageTime`

- One-hot encoding used for categorical variables

---

## Models Used

Three models were trained and evaluated:

1. Logistic Regression  
2. Random Forest  
3. XGBoost  

---

## Model Performance (Test Set)

| Model               | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------------------|----------|----------|--------|----------|--------|
| Logistic Regression| 0.8731   | 0.5961   | 0.7397 | 0.6602   | 0.8856 |
| Random Forest      | 0.8694   | 0.5851   | 0.7445 | 0.6552   | 0.8912 |
| XGBoost            | 0.8642   | 0.5683   | 0.7689 | 0.6536   | 0.8943 |

### Key Observations
- All models perform similarly in terms of F1-score (~0.65–0.66)
- XGBoost achieves the highest recall (better at detecting purchases)
- Logistic Regression provides strong baseline performance
- Random Forest achieves the highest AUC-ROC among the models

---

## 🔑 Feature Importance

Top contributing features:
- PageValue (most significant predictor)
- ExitRate
- ProductPageTime
- BounceRate

---

## 🚀 How to Run

```bash
sh run.sh
