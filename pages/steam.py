import streamlit as st
import joblib  # Use joblib instead of pickle
import numpy as np

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="AI Divorce Prediction", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_divorce_predictor.pkl")  # Load with joblib

model = load_model()  # Load model once

def main():
    st.title("🤖 Artificial Intelligence Divorce Prediction")

    st.write("### Answer the following questions:")

    questions = [
        "I can insult my spouse during our discussions.",
        "We're just starting a discussion before I know what's going on.",
        "I can be humiliating when we are discussing.",
        "The time I spent with my spouse is special for us.",
        "We share the same views about being happy in our life.",
        "I hate my spouse's way of opening a subject.",
        "My spouse and I have similar values in trust.",
        "I think that one day in the future when I look back, I will see that my spouse and I had problems.",
        "My spouse and I have similar ideas about how roles should be in marriage.",
        "I enjoy traveling with my spouse."
    ]
    
    options = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    answers = []

    # Collect user responses
    for i, question in enumerate(questions, 1):
        response = st.radio(f"Q{i}. {question}", options, index=2, key=f"q{i}")
        answers.append(options.index(response))  # Convert to numeric value

    # Predict probability
    if st.button("Submit Prediction"):
        input_data = np.array(answers).reshape(1, -1)  # Reshape for model
        
        try:
            probability = model.predict_proba(input_data)[:, 1][0]  # Get probability
            prediction_percentage = np.round(probability * 100, 2)  # Convert to percentage

            # Display result
            st.subheader("🔍 Prediction Result")
            st.info(f"**The probability of divorce is: `{prediction_percentage}%`**")

            if probability >= 0.5:
                st.error("💔 High chance of divorce. Consider seeking help.")
            else:
                st.success("❤️ Low chance of divorce. Keep nurturing your relationship!")

        except AttributeError:
            st.error("⚠️ Error: Model is not loaded correctly. Ensure it is a scikit-learn model with `predict_proba()`.")

if __name__ == "__main__":
    main()
