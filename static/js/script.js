// =========================================================
// LOANGUARD AI
// Loan Approval Prediction + AI Explanation
// =========================================================


// =========================================================
// GET HTML ELEMENTS
// =========================================================

const form = document.getElementById("loanForm");

const analyzeBtn = document.getElementById("predictBtn");
const btnText = document.getElementById("btnText");
const btnArrow = document.getElementById("btnArrow");
const btnLoader = document.getElementById("btnLoader");
const resetBtn = document.getElementById("resetBtn");

const resultSection = document.getElementById("resultSection");
const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const resultMessage = document.getElementById("resultMessage");

const probability = document.getElementById("probability");
const progressFill = document.getElementById("progressFill");


// =========================================================
// AI EXPLANATION ELEMENTS
// =========================================================

const explanationBox =
    document.getElementById("explanationBox");

const explanationList =
    document.getElementById("explanationList");


// =========================================================
// INITIAL STATE
// =========================================================

// Result is hidden when the page first loads.
resultSection.classList.add("hidden");


// =========================================================
// DISPLAY AI EXPLANATIONS
// =========================================================

function displayExplanations(explanations) {

    // Clear previous explanations
    explanationList.innerHTML = "";


    // -----------------------------------------------------
    // CHECK WHETHER EXPLANATIONS EXIST
    // -----------------------------------------------------

    if (
        !Array.isArray(explanations) ||
        explanations.length === 0
    ) {

        const emptyMessage =
            document.createElement("div");

        emptyMessage.className =
            "explanation-loading";

        emptyMessage.textContent =
            "No explanation factors are available for this prediction.";

        explanationList.appendChild(emptyMessage);

        return;
    }


    // -----------------------------------------------------
    // CREATE EXPLANATION ITEMS
    // -----------------------------------------------------

    explanations.forEach(function (item, index) {

        const explanationItem =
            document.createElement("div");

        explanationItem.className =
            "explanation-item";


        // Add positive / negative class
        if (item.impact === "Supports Approval") {

            explanationItem.classList.add("positive");

        } else {

            explanationItem.classList.add("negative");

        }


        // -------------------------------------------------
        // FACTOR ICON
        // -------------------------------------------------

        const factorIcon =
            document.createElement("div");

        factorIcon.className =
            "factor-icon";

        factorIcon.textContent =
            item.impact === "Supports Approval"
                ? "↑"
                : "↓";


        // -------------------------------------------------
        // FACTOR CONTENT
        // -------------------------------------------------

        const factorContent =
            document.createElement("div");

        factorContent.className =
            "factor-content";


        // Feature name
        const factorName =
            document.createElement("div");

        factorName.className =
            "factor-name";

        factorName.textContent =
            item.feature || "Unknown Factor";


        // Impact
        const factorImpact =
            document.createElement("div");

        factorImpact.className =
            "factor-impact";

        factorImpact.textContent =
            item.impact || "Unknown Impact";


        // -------------------------------------------------
        // BUILD ITEM
        // -------------------------------------------------

        factorContent.appendChild(factorName);

        factorContent.appendChild(factorImpact);

        explanationItem.appendChild(factorIcon);

        explanationItem.appendChild(factorContent);


        // -------------------------------------------------
        // ADD ANIMATION DELAY
        // -------------------------------------------------

        explanationItem.style.animationDelay =
            `${index * 0.08}s`;


        explanationList.appendChild(
            explanationItem
        );

    });
}


// =========================================================
// FORM SUBMIT
// =========================================================

form.addEventListener("submit", async function (event) {

    event.preventDefault();


    // =====================================================
    // LOADING STATE
    // =====================================================

    analyzeBtn.disabled = true;

    btnText.textContent =
        "Analyzing Application...";

    btnArrow.style.display =
        "none";

    btnLoader.style.display =
        "inline-block";


    // =====================================================
    // SHOW EXPLANATION LOADING STATE
    // =====================================================

    if (explanationList) {

        explanationList.innerHTML = "";

        const loadingMessage =
            document.createElement("div");

        loadingMessage.className =
            "explanation-loading";

        loadingMessage.textContent =
            "Analyzing prediction factors...";

        explanationList.appendChild(
            loadingMessage
        );
    }


    // =====================================================
    // COLLECT FORM DATA
    // =====================================================

    const data = {

        person_age:
            document.getElementById("person_age").value,

        person_gender:
            document.getElementById("person_gender").value,

        person_education:
            document.getElementById("person_education").value,

        person_income:
            document.getElementById("person_income").value,

        person_emp_exp:
            document.getElementById("person_emp_exp").value,

        person_home_ownership:
            document.getElementById("person_home_ownership").value,

        loan_amnt:
            document.getElementById("loan_amnt").value,

        loan_intent:
            document.getElementById("loan_intent").value,

        loan_int_rate:
            document.getElementById("loan_int_rate").value,

        loan_percent_income:
            document.getElementById("loan_percent_income").value,

        cb_person_cred_hist_length:
            document.getElementById(
                "cb_person_cred_hist_length"
            ).value,

        credit_score:
            document.getElementById("credit_score").value,

        previous_loan_defaults_on_file:
            document.getElementById(
                "previous_loan_defaults_on_file"
            ).value
    };


    // =====================================================
    // SEND DATA TO FLASK
    // =====================================================

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        // =================================================
        // CONVERT RESPONSE TO JSON
        // =================================================

        const result =
            await response.json();


        // =================================================
        // CHECK RESPONSE
        // =================================================

        if (!response.ok) {

            throw new Error(
                result.error ||
                "Prediction failed."
            );

        }


        // =================================================
        // SHOW RESULT SECTION
        // =================================================

        resultSection.classList.remove(
            "hidden"
        );


        // Remove previous result state

        resultSection.classList.remove(
            "approved",
            "rejected"
        );


        // Restart result animation

        void resultSection.offsetWidth;


        // =================================================
        // APPROVED
        // =================================================

        if (Number(result.prediction) === 1) {

            resultSection.classList.add(
                "approved"
            );

            resultIcon.textContent =
                "✓";

            resultTitle.textContent =
                "Loan Approved";

            resultMessage.textContent =
                "Based on the provided information, the AI model predicts that this loan application is likely to be approved.";

        }


        // =================================================
        // REJECTED
        // =================================================

        else {

            resultSection.classList.add(
                "rejected"
            );

            resultIcon.textContent =
                "×";

            resultTitle.textContent =
                "Loan Rejected";

            resultMessage.textContent =
                "Based on the provided information, the AI model predicts that this loan application is likely to be rejected.";

        }


        // =================================================
        // APPROVAL PROBABILITY
        // =================================================

        const probabilityValue =
            Number(
                result.approval_probability
            );


        probability.textContent =
            probabilityValue.toFixed(2) + "%";


        // Reset progress bar

        progressFill.style.width =
            "0%";


        // Animate progress bar

        setTimeout(function () {

            progressFill.style.width =
                Math.max(
                    0,
                    Math.min(
                        100,
                        probabilityValue
                    )
                ) + "%";

        }, 100);


        // =================================================
        // AI EXPLANATION
        // =================================================

        displayExplanations(
            result.explanations
        );


        // =================================================
        // SCROLL TO RESULT
        // =================================================

        setTimeout(function () {

            resultSection.scrollIntoView({

                behavior: "smooth",

                block: "center"

            });

        }, 150);

    }


    // =====================================================
    // ERROR HANDLING
    // =====================================================

    catch (error) {

        // Remove result states

        resultSection.classList.remove(
            "approved",
            "rejected"
        );


        // Show result section

        resultSection.classList.remove(
            "hidden"
        );


        // Restart animation

        void resultSection.offsetWidth;


        // Error state

        resultSection.classList.add(
            "rejected"
        );


        resultIcon.textContent =
            "!";


        resultTitle.textContent =
            "Prediction Error";


        resultMessage.textContent =
            error.message;


        probability.textContent =
            "--";


        progressFill.style.width =
            "0%";


        // -------------------------------------------------
        // ERROR EXPLANATION
        // -------------------------------------------------

        if (explanationList) {

            explanationList.innerHTML = "";

            const errorMessage =
                document.createElement("div");

            errorMessage.className =
                "explanation-loading";

            errorMessage.textContent =
                "AI explanation is unavailable because the prediction could not be completed.";

            explanationList.appendChild(
                errorMessage
            );
        }


        // -------------------------------------------------
        // SCROLL TO ERROR
        // -------------------------------------------------

        setTimeout(function () {

            resultSection.scrollIntoView({

                behavior: "smooth",

                block: "center"

            });

        }, 150);

    }


    // =====================================================
    // RESTORE ANALYZE BUTTON
    // =====================================================

    finally {

        analyzeBtn.disabled =
            false;


        btnText.textContent =
            "Analyze Loan Application";


        btnArrow.style.display =
            "inline-block";


        btnLoader.style.display =
            "none";
    }

});


// =========================================================
// RESET BUTTON
// =========================================================

resetBtn.addEventListener(
    "click",
    function () {


        // =================================================
        // RESET FORM
        // =================================================

        form.reset();


        // =================================================
        // HIDE RESULT
        // =================================================

        resultSection.classList.add(
            "hidden"
        );


        // Remove result states

        resultSection.classList.remove(
            "approved",
            "rejected"
        );


        // =================================================
        // RESET RESULT CONTENT
        // =================================================

        resultIcon.textContent =
            "?";


        resultTitle.textContent =
            "Awaiting Application";


        resultMessage.textContent =
            "Complete the application form and let our machine learning model analyze your loan profile.";


        probability.textContent =
            "0%";


        progressFill.style.width =
            "0%";


        // =================================================
        // RESET AI EXPLANATION
        // =================================================

        if (explanationList) {

            explanationList.innerHTML = "";

            const resetMessage =
                document.createElement("div");

            resetMessage.className =
                "explanation-loading";

            resetMessage.textContent =
                "Complete the application to see AI prediction factors.";

            explanationList.appendChild(
                resetMessage
            );
        }


        // =================================================
        // RESET ANALYZE BUTTON
        // =================================================

        analyzeBtn.disabled =
            false;


        btnText.textContent =
            "Analyze Loan Application";


        btnArrow.style.display =
            "inline-block";


        btnLoader.style.display =
            "none";


        // =================================================
        // SCROLL TO TOP
        // =================================================

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    }
);