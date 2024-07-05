import streamlit as st
import joblib
import numpy as np
import imblearn

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('klasifikasi_obesitas.pkl')
    return model

# Function to predict
def predict_obesity_level(model, height, weight, bmi, gender_num):
    input_data = np.array([[height, weight, bmi, gender_num]])
    prediction = model.predict(input_data)
    return prediction

# Main Streamlit app
def main():
    st.title('Obesity Level Classification')
    model = load_model()

    # Input fields
    height = st.number_input('Height (in meters)', min_value=0.5, max_value=2.5, value=1.75)
    weight = st.number_input('Weight (in kg)', min_value=20, max_value=200, value=70)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    gender = st.selectbox('Gender', ('Female', 'Male'))

    # Convert gender to numeric
    gender_num = 0 if gender == 'Female' else 1

    if st.button('Predict'):
        st.write("Button clicked.")
        try:
            prediction = predict_obesity_level(model, height, weight, bmi, gender_num)
            obesity_level = ['Extremely Weak','Weak','Normal', 'Overweight', 'Obese','Extremely Obese']
            st.write(f'The predicted obesity level is: {obesity_level[prediction[0]]}')
        except Exception as e:
            st.write(f'Error during prediction: {e}')

if __name__ == '__main__':
    main()
