from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure you have the scaler saved

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    input_data = [float(request.form[feature]) for feature in features]
    
    # Scale the numerical features
    numerical_features = input_data[:3] + input_data[4:8] + input_data[9:]
    scaled_numerical = scaler.transform([numerical_features])
    
    # Combine with categorical features
    final_input = np.concatenate((
        scaled_numerical[0], 
        [input_data[3]],   # CHAS
        # [input_data[7]],   # DIS
        [input_data[8]] # RAD
        # [input_data[-1]]   # LSTAT
    )).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_input)[0]
    
    return render_template('frontend.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)