
import streamlit as st
import pandas as pd
import joblib
import os

# --- Load the trained model and label encoders ---
# Ensure these files are in the same directory as your Streamlit app.py or accessible via path
model_path = 'best_model.pkl'
encoders_path = 'label_encoders(1).pkl'

if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}")
    st.stop()

if not os.path.exists(encoders_path):
    st.error(f"Error: Label encoders file not found at {encoders_path}")
    st.stop()

try:
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
except Exception as e:
    st.error(f"Error loading model or label encoders: {e}")
    st.stop()

# --- Streamlit UI ---
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields
age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
education_level = st.selectbox('Education Level', label_encoders['Education Level'].classes_)
job_title = st.selectbox('Job Title', label_encoders['Job Title'].classes_)
years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, 0.5)

# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame for the input features
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education_level],
        'Job Title': [job_title],
        'Years of Experience': [years_of_experience]
    })

    # Apply Label Encoding to categorical features using the loaded encoders
    for col in ['Gender', 'Education Level', 'Job Title']:
        if col in input_df.columns and col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding '{col}': {e}. Please ensure the selected category is valid.")
                st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_df)
        st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Error making prediction: {e}")
