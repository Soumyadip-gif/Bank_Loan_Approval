import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/loan_data.csv")

print("Dataset Shape:", df.shape)


# ==========================================================
# 2. REMOVE INVALID RECORDS
# ==========================================================

df = df[
    (df["person_age"] <= 80) &
    (df["person_emp_exp"] <= 60)
]

print("\nAfter removing invalid records:")
print("Dataset Shape:", df.shape)


# ==========================================================
# 3. MISSING VALUES
# ==========================================================

print("\n--- Missing Values ---")
print(df.isnull().sum())


# ==========================================================
# 4. DUPLICATE ROWS
# ==========================================================

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())


# ==========================================================
# 5. DATA TYPES
# ==========================================================

print("\n--- Data Types ---")
print(df.dtypes)


# ==========================================================
# 6. TARGET DISTRIBUTION
# ==========================================================

print("\n--- Loan Status Distribution ---")
print(df["loan_status"].value_counts())


# ==========================================================
# 7. NUMERICAL STATISTICS
# ==========================================================

print("\n--- Numerical Statistics ---")
print(df.describe())


# ==========================================================
# 8. SUSPICIOUS VALUE CHECKS
# ==========================================================

print("\n--- Age > 100 ---")
print(df[df["person_age"] > 100][["person_age"]])

print("\n--- Employment Experience > 50 ---")
print(
    df[df["person_emp_exp"] > 50][
        ["person_age", "person_emp_exp"]
    ]
)

print("\n--- Very High Income ---")
print(
    df[df["person_income"] > 1_000_000][
        ["person_age", "person_income"]
    ]
)


# ==========================================================
# 9. SEPARATE FEATURES AND TARGET
# ==========================================================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())


# ==========================================================
# 10. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# ==========================================================
# 11. DEFINE FEATURES
# ==========================================================

categorical_features = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

numerical_features = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]


# ==========================================================
# 12. PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("\n--- Preprocessor Created Successfully ---")
print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))


# ==========================================================
# 13. LOGISTIC REGRESSION
# ==========================================================

print("\n--- Training Logistic Regression ---")

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("Logistic Regression Training Completed!")

print("\n==========================================")
print("LOGISTIC REGRESSION RESULTS")
print("==========================================")

print(
    "Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_pred
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)


# ==========================================================
# 14. RANDOM FOREST
# ==========================================================

print("\n--- Training Random Forest ---")

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("Random Forest Training Completed!")

print("\n==========================================")
print("RANDOM FOREST RESULTS")
print("==========================================")

print(
    "Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)


# ==========================================================
# 15. FEATURE IMPORTANCE
# ==========================================================

print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

# Get the trained preprocessor from the Random Forest pipeline
trained_preprocessor = rf_model.named_steps["preprocessor"]

# Get the trained Random Forest classifier
trained_classifier = rf_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = trained_preprocessor.get_feature_names_out()

# Get Random Forest feature importance
importance_values = trained_classifier.feature_importances_

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort from highest to lowest importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# Display top 15 features
print("\nTop 15 Important Features:")

print(
    feature_importance.head(15).to_string(index=False)
)


# ==========================================================
# 16. SAVE FEATURE IMPORTANCE
# ==========================================================

model_directory = "model"

os.makedirs(
    model_directory,
    exist_ok=True
)

feature_importance_path = os.path.join(
    model_directory,
    "feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)

print("\n==========================================")
print("FEATURE IMPORTANCE SAVING")
print("==========================================")

print("Feature Importance Saved Successfully!")
print("Location:", feature_importance_path)


# ==========================================================
# 17. MODEL COMPARISON
# ==========================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)


# ==========================================================
# 18. SAVE FINAL MODEL
# ==========================================================

model_path = os.path.join(
    model_directory,
    "loan_model.pkl"
)

joblib.dump(
    rf_model,
    model_path
)

print("\n==========================================")
print("MODEL SAVING")
print("==========================================")

print("Final Model: Random Forest")
print("Model Saved Successfully!")
print("Location:", model_path)