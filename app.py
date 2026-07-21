import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_rf = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_rf = None


# ==========================================================
# Prediction Function
# ==========================================================
def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    # Empty input validation
    inputs = [
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ]

    if any(v is None for v in inputs):
        return "❌ Please fill in all fields."

    # Type conversion
    try:
        no_of_dependents = int(no_of_dependents)
        education = int(education)
        self_employed = int(self_employed)
        income_annum = float(income_annum)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)
        cibil_score = int(cibil_score)
        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)

    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."

    # Negative value validation
    numeric_values = [
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ]

    if any(v < 0 for v in numeric_values):
        return "❌ Negative values are not allowed."

    # Additional validations
    if not (300 <= cibil_score <= 900):
        return "❌ CIBIL score must be between 300 and 900."

    if no_of_dependents > 20:
        return "❌ Number of dependents seems unusually high (Maximum 20)."

    if income_annum <= 0:
        return "❌ Annual income must be greater than zero."

    if loan_amount <= 0:
        return "❌ Loan amount must be greater than zero."

    if loan_term <= 0:
        return "❌ Loan term must be greater than zero."

    # Model check
    if deployed_rf is None:
        return "❌ Model failed to load. Please check loan_prediction_model.pkl."

    # Prediction
    try:
        input_data = [[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
        ]]

        prediction = deployed_rf.predict(input_data)

        # Change this if your model uses opposite labels
        if prediction[0] == 0:
            return (
                "🟢 Prediction Result\n\n"
                "Loan Status: APPROVED\n\n"
                "The applicant meets the criteria for this loan."
            )
        else:
            return (
                "🔴 Prediction Result\n\n"
                "Loan Status: REJECTED\n\n"
                "The applicant does not meet the criteria."
            )

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"


# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 🏦 Loan Approval Prediction System

This application predicts whether an applicant's loan will be **Approved** or **Rejected**
using a trained **Random Forest Machine Learning Model**.

Enter the applicant's financial and personal details below.
"""

developer_info = """
### About the Developer

**Created by:** Sachin

---

### 🛠️ Tools & Technologies Used

- Machine Learning: Scikit-learn (Random Forest Classifier)
- Web Framework: Gradio
- Language: Python
- Deployment: Render
"""


# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Dropdown(
            choices=[("Graduate", 1), ("Not Graduate", 0)],
            label="Education Status"
        ),
        gr.Dropdown(
            choices=[("Yes", 1), ("No", 0)],
            label="Self Employed?"
        ),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount Requested"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score (300-900)"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Assessment Result", lines=6),
    title="🏦 Loan Approval Prediction System",
    description=DESCRIPTION,
    article=developer_info,
)


# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
