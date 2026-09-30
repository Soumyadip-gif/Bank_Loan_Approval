import pandas as pd
import joblib
import shap


# ==========================================================
# 1. LOAD TRAINED MODEL
# ==========================================================

model = joblib.load("model/loan_model.pkl")

print("Loan Model Loaded Successfully!")


# ==========================================================
# 2. CREATE SAMPLE APPLICATION
# ==========================================================

sample_data = pd.DataFrame([{
    "person_age": 25,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_income": 60000,
    "person_emp_exp": 3,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4,
    "credit_score": 680,
    "previous_loan_defaults_on_file": "No"
}])


# ==========================================================
# 3. MAKE PREDICTION
# ==========================================================

prediction = model.predict(sample_data)[0]

probability = model.predict_proba(
    sample_data
)[0]

approval_probability = probability[1] * 100


print("\n==========================================")
print("LOAN PREDICTION")
print("==========================================")

if prediction == 1:
    print("Prediction: Approved")
else:
    print("Prediction: Rejected")

print(
    f"Approval Probability: {approval_probability:.2f}%"
)


# ==========================================================
# 4. GET PREPROCESSOR AND CLASSIFIER
# ==========================================================

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]


# ==========================================================
# 5. TRANSFORM APPLICATION DATA
# ==========================================================

transformed_data = preprocessor.transform(
    sample_data
)


# ==========================================================
# 6. GET FEATURE NAMES
# ==========================================================

feature_names = preprocessor.get_feature_names_out()


# ==========================================================
# 7. CREATE SHAP EXPLAINER
# ==========================================================

explainer = shap.TreeExplainer(
    classifier
)


# ==========================================================
# 8. CALCULATE SHAP VALUES
# ==========================================================

shap_values = explainer.shap_values(
    transformed_data
)


# ==========================================================
# 9. HANDLE SHAP OUTPUT
# ==========================================================

if isinstance(shap_values, list):

    values = shap_values[1][0]

else:

    values = shap_values[0]

    if values.ndim > 1:
        values = values[:, 1]


# ==========================================================
# 10. CREATE EXPLANATION TABLE
# ==========================================================

explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": values
})

explanation["Impact"] = (
    explanation["SHAP_Value"].abs()
)

explanation = explanation.sort_values(
    "Impact",
    ascending=False
)


# ==========================================================
# 11. DISPLAY TOP 10 FEATURES
# ==========================================================

print("\n==========================================")
print("TOP 10 PREDICTION FACTORS")
print("==========================================")

print(
    explanation[
        [
            "Feature",
            "SHAP_Value"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================================
# 12. DISPLAY POSITIVE / NEGATIVE IMPACT
# ==========================================================

print("\n==========================================")
print("FACTOR IMPACT")
print("==========================================")

for _, row in explanation.head(10).iterrows():

    feature = row["Feature"]
    shap_value = row["SHAP_Value"]

    if shap_value > 0:

        impact = "Supports Approval"

    else:

        impact = "Supports Rejection"

    print(
        f"{feature}: "
        f"{impact} "
        f"({shap_value:.4f})"
    )