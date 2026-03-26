# ===============================
# LOAN RISK INTELLIGENCE SYSTEM (FINAL)
# ===============================

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

np.random.seed(42)

# ===============================
# 1. DATA GENERATION
# ===============================
n = 1000

df = pd.DataFrame({
    "Applicant_Income": np.random.lognormal(11, 0.45, n).astype(int),
    "Coapplicant_Income": np.random.lognormal(10.5, 0.5, n).astype(int),
    "Credit_Score": np.random.normal(680, 100, n).clip(300, 900).astype(int),
    "Loan_Amount": np.random.lognormal(12, 0.55, n).astype(int),
    "Loan_Term": np.random.choice([12,24,36,60,120,180,240,360], n)
})

# ===============================
# 2. FEATURE ENGINEERING
# ===============================
df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
df["Debt_Income_Ratio"] = df["Loan_Amount"] / (df["Total_Income"] + 1)
df["Monthly_Payment"] = df["Loan_Amount"] / df["Loan_Term"]

# ===============================
# 3. TARGET GENERATION
# ===============================
risk = (
    0.4*(df["Credit_Score"]/900) +
    0.25*(df["Total_Income"]/df["Loan_Amount"]) +
    0.2*(1/(df["Debt_Income_Ratio"]+0.5)) +
    0.15*(1/(df["Loan_Term"]/360+0.01))
)

prob = risk - 0.2*(df["Debt_Income_Ratio"]>3) - 0.15*(df["Credit_Score"]<550)
df["Approval_Status"] = (prob > 0.75).astype(int)

# ===============================
# 4. KMEANS SEGMENTATION (NOVEL)
# ===============================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["Segment"] = kmeans.fit_predict(
    df[["Credit_Score","Debt_Income_Ratio","Total_Income","Loan_Amount"]]
)

labels = {0:"High Risk",1:"Prime Borrower",2:"Moderate Risk",3:"Emerging Credit"}
df["Segment_Name"] = df["Segment"].map(labels)

# ===============================
# 5. ENCODING
# ===============================
bins = [0,600,700,800,900]
df["Credit_Category"] = pd.cut(df["Credit_Score"], bins=bins)

le = LabelEncoder()
df["Credit_Category_Enc"] = le.fit_transform(df["Credit_Category"].astype(str))

# ===============================
# 6. MODEL
# ===============================
features = [
    "Applicant_Income","Coapplicant_Income","Credit_Score",
    "Loan_Amount","Loan_Term","Total_Income",
    "Debt_Income_Ratio","Monthly_Payment","Credit_Category_Enc"
]

X = df[features]
y = df["Approval_Status"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000).fit(X_train,y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200,max_depth=10).fit(X_train,y_train)

# GridSearch
grid = GridSearchCV(LogisticRegression(max_iter=1000), {"C":[0.01,0.1,1,10]}, cv=5)
grid.fit(X_train,y_train)
best = grid.best_estimator_

# ===============================
# 7. METRICS
# ===============================
pred = best.predict(X_test)
prob = best.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test,pred)
roc = roc_auc_score(y_test,prob)
cm = confusion_matrix(y_test,pred)

# ===============================
# 8. EXPORT FOR FRONTEND
# ===============================
output = {
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "lr_coef": best.coef_[0].tolist(),
    "lr_intercept": float(best.intercept_[0]),
    "accuracy": float(accuracy),
    "roc_auc": float(roc),
    "confusion_matrix": cm.tolist(),
    "segment_counts": df["Segment_Name"].value_counts().to_dict(),
    "segment_approval_rates": df.groupby("Segment_Name")["Approval_Status"].mean().to_dict(),
    "class_dist": {
        "Approved": int(df["Approval_Status"].sum()),
        "Rejected": int((df["Approval_Status"]==0).sum())
    }
}

with open("C:\\Users\\steve\\Downloads\\Predictive Project\\frontend\\src\\data\\simulator_data.json","w") as f:
    json.dump(output,f,indent=2)

print("✅ Backend ready. JSON exported.")