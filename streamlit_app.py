import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
with open('student_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("🎓 EduPredict")

# Input fields
hours_studied = st.number_input("📚 Hours Studied", min_value=0.0, step=0.5)
prev_score = st.number_input("📊 Previous Year Score (out of 100)", min_value=0.0, max_value=100.0)
extra_activities = st.selectbox("🏃 Participated in Extra Activities?", ["No", "Yes"])
sleep_hours = st.number_input("💤 Sleep Hours", min_value=0.0, step=0.5)
sample_papers = st.number_input("📝 Number of Sample Papers Solved", min_value=0)

# Encode the categorical input
extra_activities_encoded = 1 if extra_activities == "Yes" else 0

# Prediction logic

if st.button("🎯 Predict Performance"):
    input_data = np.array([[hours_studied, prev_score, extra_activities_encoded, sleep_hours, sample_papers]])
    prediction = model.predict(input_data)[0]

    st.success(f"📈 Predicted Score: {prediction:.2f}")

    # Bar graph
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(['Score'], [prediction], color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title("Predicted Performance")
    st.pyplot(fig)

    # CSV download (moved inside the same block)
    record = pd.DataFrame([{
        "Hours Studied": hours_studied,
        "Previous Score": prev_score,
        "Extra Activities": extra_activities,
        "Sleep Hours": sleep_hours,
        "Sample Papers Solved": sample_papers,
        "Predicted Score": prediction
    }])

    st.download_button(
        label="📥 Download Prediction as CSV",
        data=record.to_csv(index=False),
        file_name='prediction_result.csv',
        mime='text/csv'
    )
