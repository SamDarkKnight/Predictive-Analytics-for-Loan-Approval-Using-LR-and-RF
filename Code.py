# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# ===============================
# 2. Generate Synthetic Dataset
# ===============================
n = 1000

# ===============================
# 3. Generate Applicant Profiles
# ===============================

df = pd.DataFrame({
    
    # Income follows lognormal distribution (realistic)
    "Applicant_Income": np.random.lognormal(mean=11, sigma=0.45, size=n).astype(int),
    
    # Coapplicant income usually smaller
    "Coapplicant_Income": np.random.lognormal(mean=10.5, sigma=0.5, size=n).astype(int),
    
    # Credit score normally distributed
    "Credit_Score": np.random.normal(680, 100, n).clip(300, 900).astype(int),
    
    # Loan amount correlated with income
    "Loan_Amount": np.random.lognormal(mean=12, sigma=0.55, size=n).astype(int),
    
    # Loan duration distribution
    "Loan_Term": np.random.choice(
        [12,24,36,60,120,180,240,360],
        n,
        p=[0.03,0.07,0.15,0.25,0.2,0.12,0.1,0.08]
    )
})

# ===============================
# 4. Feature Engineering
# ===============================

df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]

# Debt-to-income ratio
df["Debt_Income_Ratio"] = df["Loan_Amount"] / (df["Total_Income"] + 1)

# Monthly loan payment approximation
df["Monthly_Payment"] = df["Loan_Amount"] / df["Loan_Term"]

# ===============================
# 5. Risk Score Calculation
# ===============================

credit_factor = df["Credit_Score"] / 900
income_factor = df["Total_Income"] / df["Loan_Amount"]
term_factor = 1 / (df["Loan_Term"]/360 + 0.01)
debt_factor = 1 / (df["Debt_Income_Ratio"] + 0.5)

risk_score = (
    0.4 * credit_factor +
    0.25 * income_factor +
    0.2 * debt_factor +
    0.15 * term_factor
)

# Add real-world randomness
noise = np.random.normal(0, 0.05, n)

approval_prob = (
    risk_score
    - 0.2*(df["Debt_Income_Ratio"] > 3)
    - 0.15*(df["Credit_Score"] < 550)
    + noise
)

# ===============================
# 6. Final Loan Approval
# ===============================

df["Approval_Status"] = (approval_prob > 0.75).astype(int)

# ===============================
# 7. Simulate Missing Values
# ===============================

for col in df.columns:
    if col != "Approval_Status":
        df.loc[df.sample(frac=0.02).index, col] = np.nan
        
# ===============================
# 8. Dataset Preview
# ===============================

print("Dataset Preview:")
print(df.head())

print("\nClass Distribution:")
print(df["Approval_Status"].value_counts())

# ===============================
# 9. Identify Data Quality Issues
# ===============================

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ===============================
# 10. Handle Missing Values
# ===============================

df.fillna(df.median(), inplace=True)

print("\nStatistical Summary")
print(df.describe())

# ===============================
# 11. Remove Noisy Predictors
# (Example: none here, but check correlation)
# ===============================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()


# ===============================
# 12. Credit Score Binning
# ===============================

bins = [0,600,700,800,900]
labels = ["Poor","Average","Good","Excellent"]

df["Credit_Category"] = pd.cut(df["Credit_Score"], bins=bins, labels=labels)

encoder = LabelEncoder()
df["Credit_Category"] = encoder.fit_transform(df["Credit_Category"])

# ===============================
# 13. Prepare Data for Model
# ===============================
X = df.drop("Approval_Status", axis=1)
y = df["Approval_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 14. Train Model
# ===============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nModel Comparison")
print("----------------")
print("Logistic Regression:", accuracy_score(y_test, y_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))

# ===============================
# 15. Hyperparameter Tuning
# ===============================

params = {
    "C":[0.01,0.1,1,10,100]
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameter:", grid.best_params_)

# ===============================
# 16. Evaluate Tuned Model
# ===============================

y_pred_tuned = best_model.predict(X_test)

# ROC-AUC calculation
y_prob = best_model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nModel Performance AFTER Tuning")

importance = pd.Series(best_model.coef_[0], index=X.columns)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
print()

print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))