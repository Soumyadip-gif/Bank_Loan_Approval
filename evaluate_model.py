import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/loan_data.csv")

print("Dataset Loaded Successfully!")


# ==========================================================
# 2. REMOVE INVALID RECORDS
# ==========================================================

df = df[
    (df["person_age"] <= 80) &
    (df["person_emp_exp"] <= 60)
]

print("Invalid Records Removed!")


# ==========================================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]


# ==========================================================
# 4. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 5. LOAD FINAL MODEL
# ==========================================================

model_path = "model/loan_model.pkl"

model = joblib.load(model_path)

print("Random Forest Model Loaded Successfully!")


# ==========================================================
# 6. MAKE PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================================
# 7. CALCULATE EVALUATION METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================================
# 8. DISPLAY RESULTS
# ==========================================================

print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ==========================================================
# 9. CLASSIFICATION REPORT
# ==========================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ]
    )
)


# ==========================================================
# 10. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================================
# 11. SAVE METRICS
# ==========================================================

metrics_data = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

os.makedirs(
    "model",
    exist_ok=True
)

metrics_path = "model/model_metrics.csv"

metrics_data.to_csv(
    metrics_path,
    index=False
)


# ==========================================================
# 12. FINAL VALIDATION
# ==========================================================

print("\n==========================================")
print("FINAL VALIDATION")
print("==========================================")

if os.path.exists(model_path):

    print("✓ Model File Found")

else:

    print("✗ Model File Missing")


if accuracy >= 0.90:

    print("✓ Accuracy Check Passed")

else:

    print("⚠ Accuracy Below 90%")


if len(y_pred) == len(y_test):

    print("✓ Prediction Count Check Passed")

else:

    print("✗ Prediction Count Check Failed")


print("\nMetrics Saved Successfully!")

print(
    "Location:",
    metrics_path
)

print("\n==========================================")
print("STEP 20 COMPLETED")
print("==========================================")