import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD FEATURE IMPORTANCE
# ==========================================================

file_path = "model/feature_importance.csv"

df = pd.read_csv(file_path)

print("Feature Importance File Loaded Successfully!")


# ==========================================================
# 2. SELECT TOP 10 FEATURES
# ==========================================================

top_features = df.head(10).copy()

# Remove preprocessing prefixes
top_features["Feature"] = (
    top_features["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# Sort for horizontal bar chart
top_features = top_features.sort_values(
    by="Importance",
    ascending=True
)


# ==========================================================
# 3. CREATE VISUALIZATION
# ==========================================================

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Features Influencing Loan Approval")

plt.tight_layout()


# ==========================================================
# 4. SAVE CHART
# ==========================================================

os.makedirs("model", exist_ok=True)

output_path = "model/feature_importance.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\n==========================================")
print("FEATURE IMPORTANCE VISUALIZATION")
print("==========================================")

print("Top 10 Features:")
print(top_features.to_string(index=False))

print("\nChart Saved Successfully!")
print("Location:", output_path)


# ==========================================================
# 5. DISPLAY CHART
# ==========================================================

plt.show()