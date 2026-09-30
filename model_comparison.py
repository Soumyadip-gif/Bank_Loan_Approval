import os
import matplotlib.pyplot as plt


# ==========================================================
# 1. MODEL ACCURACY VALUES
# ==========================================================

models = [
    "Logistic Regression",
    "Random Forest"
]

accuracies = [
    85.17,
    92.58
]


# ==========================================================
# 2. CREATE BAR CHART
# ==========================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    accuracies
)


# ==========================================================
# 3. ADD ACCURACY VALUES
# ==========================================================

for bar, accuracy in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )


# ==========================================================
# 4. CHART DETAILS
# ==========================================================

plt.title(
    "Loan Approval Model Performance Comparison",
    fontsize=14
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ==========================================================
# 5. SAVE CHART
# ==========================================================

os.makedirs(
    "model",
    exist_ok=True
)

output_path = "model/model_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("==========================================")
print("MODEL PERFORMANCE VISUALIZATION")
print("==========================================")

print("Chart Saved Successfully!")
print("Location:", output_path)


# ==========================================================
# 6. DISPLAY CHART
# ==========================================================

plt.show()