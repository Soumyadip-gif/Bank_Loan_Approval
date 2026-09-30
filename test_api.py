import requests


url = "http://127.0.0.1:5000/predict"


data = {
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
}


response = requests.post(
    url,
    json=data
)


print("Status Code:", response.status_code)

print("\nResponse:")

print(response.json())