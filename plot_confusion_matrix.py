import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
# 5. LOAD TRAINED MODEL
# ==========================================================

model = joblib.load("model/loan_model.pkl")

print("Trained Random Forest Model Loaded Successfully!")


# ==========================================================
# 6. MAKE TEST PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)


# ==========================================================
# 7. CREATE CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================================
# 8. DISPLAY CONFUSION MATRIX
# ==========================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Rejected",
        "Approved"
    ]
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Random Forest - Loan Approval Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()


# ==========================================================
# 9. SAVE CONFUSION MATRIX
# ==========================================================

os.makedirs(
    "model",
    exist_ok=True
)

output_path = "model/confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\n==========================================")
print("CONFUSION MATRIX VISUALIZATION")
print("==========================================")

print("Chart Saved Successfully!")
print("Location:", output_path)


# ==========================================================
# 10. DISPLAY CHART
# ==========================================================

plt.show()