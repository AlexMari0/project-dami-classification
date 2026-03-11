from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing tools
with open('../data-mining-project/notebooks/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../data-mining-project/notebooks/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../data-mining-project/notebooks/ohe_columns.pkl', 'rb') as f:
    ohe_columns = pickle.load(f)

with open('../data-mining-project/notebooks/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect the user's input from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    city = request.form['city']
    working_status = request.form['working_status']
    profession = request.form['profession']
    academic_pressure = float(request.form['academic_pressure'])
    work_pressure = float(request.form['work_pressure'])
    cgpa = float(request.form['cgpa'])
    study_satisfaction = float(request.form['study_satisfaction'])
    job_satisfaction = float(request.form['job_satisfaction'])
    sleep_duration = request.form['sleep_duration']
    dietary_habits = request.form['dietary_habits']
    degree = request.form['degree']
    suicidal_thoughts = request.form['suicidal_thoughts']
    work_study_hours = float(request.form['work_study_hours'])
    financial_stress = float(request.form['financial_stress'])
    mental_illness_history = request.form['mental_illness_history']

    # Separate numerical and categorical features
    numerical_features = [
        age, academic_pressure, work_pressure, cgpa,
        study_satisfaction, job_satisfaction, work_study_hours, financial_stress
    ]
    categorical_features = [
        gender, city, working_status, profession, sleep_duration,
        dietary_habits, degree, suicidal_thoughts, mental_illness_history
    ]

    # Preprocess the input data
    input_data = preprocess_data(numerical_features, categorical_features, scaler, ohe_columns, feature_names)

    # Debug: Print the input data
    print("Processed Input Data:", input_data)

# Make the prediction
    prediction = model.predict(input_data)
    print("Model Prediction:", prediction)

    # Use a threshold to convert probabilities to classes
    if prediction[0][0] >= 0.5:
        result = 'You might be depressed'
    else:
        result = 'You are not depressed'
        
    return render_template('index.html', result=result)

def preprocess_data(numerical_features, categorical_features, scaler, ohe_columns, feature_names):
    # Convert numerical features to DataFrame with proper column names for the scaler
    numerical_features_df = pd.DataFrame([numerical_features], columns=scaler.feature_names_in_)
    numerical_features_scaled = scaler.transform(numerical_features_df)

    # Create a DataFrame for categorical features with dummy variables
    categorical_df = pd.DataFrame(0, index=[0], columns=ohe_columns)

    # Set relevant columns to 1 based on input categories
    for category in categorical_features:
        col_name = f"{category}"  # Replace with the correct prefix logic if needed
        if col_name in categorical_df.columns:
            categorical_df[col_name] = 1

    # Debugging: Print categorical data
    print("Categorical DataFrame:\n", categorical_df)

    # Combine numerical and categorical features
    combined_features = np.hstack([numerical_features_scaled, categorical_df.to_numpy()])

    # Convert to DataFrame and ensure all feature columns are present for the model
    input_df = pd.DataFrame(combined_features, columns=feature_names)

    # Add missing columns to align with the model's feature names
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the model's expected feature order
    input_df = input_df[feature_names]

    return input_df.to_numpy()

if __name__ == '__main__':
    app.run(debug=True)
