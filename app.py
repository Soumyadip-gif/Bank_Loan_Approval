# =========================================================
# LOANGUARD AI
# Loan Approval Prediction
# Flask Backend + SHAP Explainability
# =========================================================


# =========================================================
# IMPORT LIBRARIES
# =========================================================

from flask import Flask, request, jsonify, render_template

import pandas as pd
import joblib
import shap


# =========================================================
# INITIALIZE FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(
    "model/loan_model.pkl"
)


# =========================================================
# GET MODEL COMPONENTS
# =========================================================

preprocessor = model.named_steps[
    "preprocessor"
]

classifier = model.named_steps[
    "classifier"
]


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(
    classifier
)


print(
    "Loan Model Loaded Successfully!"
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# LOAN PREDICTION API
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # =================================================
        # GET JSON DATA
        # =================================================

        data = request.get_json()

        if not data:
            raise ValueError(
                "No application data received."
            )


        # =================================================
        # CONVERT NUMERICAL VALUES
        # =================================================

        age = float(
            data["person_age"]
        )

        income = float(
            data["person_income"]
        )

        emp_exp = int(
            data["person_emp_exp"]
        )

        loan_amount = float(
            data["loan_amnt"]
        )

        interest_rate = float(
            data["loan_int_rate"]
        )

        loan_percent_income = float(
            data["loan_percent_income"]
        )

        credit_history = float(
            data[
                "cb_person_cred_hist_length"
            ]
        )

        credit_score = int(
            data["credit_score"]
        )


        # =================================================
        # INPUT VALIDATION
        # =================================================

        if age < 20 or age > 80:

            raise ValueError(
                "Age must be between 20 and 80."
            )


        if emp_exp < 0 or emp_exp > 60:

            raise ValueError(
                "Employment experience must be between 0 and 60 years."
            )


        if income <= 0:

            raise ValueError(
                "Income must be greater than 0."
            )


        if loan_amount <= 0:

            raise ValueError(
                "Loan amount must be greater than 0."
            )


        if interest_rate < 0 or interest_rate > 100:

            raise ValueError(
                "Interest rate must be between 0 and 100."
            )


        if loan_percent_income < 0 or loan_percent_income > 1:

            raise ValueError(
                "Loan-to-income ratio must be between 0 and 1."
            )


        if credit_history < 0:

            raise ValueError(
                "Credit history length cannot be negative."
            )


        if credit_score < 300 or credit_score > 850:

            raise ValueError(
                "Credit score must be between 300 and 850."
            )


        if emp_exp > age:

            raise ValueError(
                "Employment experience cannot be greater than age."
            )


        if loan_amount > income:

            raise ValueError(
                "Loan amount cannot be greater than annual income."
            )


        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame([
            {

                "person_age":
                    age,

                "person_gender":
                    data["person_gender"],

                "person_education":
                    data["person_education"],

                "person_income":
                    income,

                "person_emp_exp":
                    emp_exp,

                "person_home_ownership":
                    data[
                        "person_home_ownership"
                    ],

                "loan_amnt":
                    loan_amount,

                "loan_intent":
                    data["loan_intent"],

                "loan_int_rate":
                    interest_rate,

                "loan_percent_income":
                    loan_percent_income,

                "cb_person_cred_hist_length":
                    credit_history,

                "credit_score":
                    credit_score,

                "previous_loan_defaults_on_file":
                    data[
                        "previous_loan_defaults_on_file"
                    ]

            }
        ])


        # =================================================
        # MODEL PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]


        # =================================================
        # PREDICTION PROBABILITY
        # =================================================

        probability = model.predict_proba(
            input_data
        )[0]


        approval_probability = (
            probability[1] * 100
        )


        # =================================================
        # RESULT
        # =================================================

        result = (
            "Approved"
            if prediction == 1
            else "Rejected"
        )


        # =================================================
        # TRANSFORM DATA FOR SHAP
        # =================================================

        transformed_data = (
            preprocessor.transform(
                input_data
            )
        )


        # =================================================
        # CALCULATE SHAP VALUES
        # =================================================

        shap_values = explainer.shap_values(
            transformed_data
        )


        # =================================================
        # HANDLE SHAP OUTPUT
        # =================================================

        if isinstance(
            shap_values,
            list
        ):

            values = shap_values[1][0]

        else:

            values = shap_values[0]

            if values.ndim > 1:

                values = values[:, 1]


        # =================================================
        # GET FEATURE NAMES
        # =================================================

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # =================================================
        # CREATE SHAP DATAFRAME
        # =================================================

        explanation = pd.DataFrame({

            "Feature":
                feature_names,

            "SHAP_Value":
                values

        })


        explanation["Impact"] = (
            explanation[
                "SHAP_Value"
            ].abs()
        )


        # =================================================
        # SORT BY IMPORTANCE
        # =================================================

        explanation = (
            explanation
            .sort_values(
                "Impact",
                ascending=False
            )
        )


        # =================================================
        # FEATURE NAME MAPPING
        # =================================================

        feature_names_map = {

            "num__person_age":
                "Age",

            "num__person_income":
                "Income",

            "num__person_emp_exp":
                "Employment Experience",

            "num__loan_amnt":
                "Loan Amount",

            "num__loan_int_rate":
                "Loan Interest Rate",

            "num__loan_percent_income":
                "Loan/Income Ratio",

            "num__cb_person_cred_hist_length":
                "Credit History Length",

            "num__credit_score":
                "Credit Score",


            "cat__person_gender_male":
                "Gender",

            "cat__person_gender_female":
                "Gender",


            "cat__person_education_High School":
                "Education",

            "cat__person_education_Bachelor":
                "Education",

            "cat__person_education_Master":
                "Education",

            "cat__person_education_Associate":
                "Education",

            "cat__person_education_Doctorate":
                "Education",


            "cat__person_home_ownership_RENT":
                "Home Ownership",

            "cat__person_home_ownership_OWN":
                "Home Ownership",

            "cat__person_home_ownership_MORTGAGE":
                "Home Ownership",

            "cat__person_home_ownership_OTHER":
                "Home Ownership",


            "cat__loan_intent_PERSONAL":
                "Loan Purpose",

            "cat__loan_intent_EDUCATION":
                "Loan Purpose",

            "cat__loan_intent_MEDICAL":
                "Loan Purpose",

            "cat__loan_intent_VENTURE":
                "Loan Purpose",

            "cat__loan_intent_HOMEIMPROVEMENT":
                "Loan Purpose",

            "cat__loan_intent_DEBTCONSOLIDATION":
                "Loan Purpose",


            "cat__previous_loan_defaults_on_file_No":
                "Previous Loan Default",

            "cat__previous_loan_defaults_on_file_Yes":
                "Previous Loan Default"
        }


        # =================================================
        # CREATE USER-FRIENDLY EXPLANATIONS
        # =================================================

        explanations = []

        used_features = set()


        for _, row in explanation.iterrows():

            raw_feature = row[
                "Feature"
            ]


            display_feature = (
                feature_names_map.get(
                    raw_feature,
                    raw_feature
                )
            )


            # ---------------------------------------------
            # Avoid duplicate user-facing features
            # ---------------------------------------------

            if display_feature in used_features:

                continue


            used_features.add(
                display_feature
            )


            # ---------------------------------------------
            # SHAP VALUE
            # ---------------------------------------------

            shap_value = float(
                row["SHAP_Value"]
            )


            # ---------------------------------------------
            # IMPACT
            # ---------------------------------------------

            if shap_value > 0:

                impact = (
                    "Supports Approval"
                )

            else:

                impact = (
                    "Supports Rejection"
                )


            # ---------------------------------------------
            # ADD EXPLANATION
            # ---------------------------------------------

            explanations.append({

                "feature":
                    display_feature,

                "impact":
                    impact,

                "shap_value":
                    round(
                        shap_value,
                        4
                    )

            })


            # ---------------------------------------------
            # TOP 5 FEATURES
            # ---------------------------------------------

            if len(explanations) == 5:

                break


        # =================================================
        # RETURN RESPONSE
        # =================================================

        return jsonify({

            "prediction":
                int(prediction),

            "result":
                result,

            "approval_probability":
                round(
                    approval_probability,
                    2
                ),

            "explanations":
                explanations

        })


    # =====================================================
    # VALIDATION ERROR
    # =====================================================

    except ValueError as e:

        return jsonify({

            "error":
                str(e)

        }), 400


    # =====================================================
    # MISSING FIELD ERROR
    # =====================================================

    except KeyError as e:

        return jsonify({

            "error":
                f"Missing field: {str(e)}"

        }), 400


    # =====================================================
    # GENERAL ERROR
    # =====================================================

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 400


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )