# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model=pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))

# List of features used by the model
FEATURES = [
    'concave points_worst', 'perimeter_worst', 'concave points_mean','radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean','area_mean', 'concavity_mean','concavity_worst', 'compactness_mean','compactness_worst', 'radius_se', 'perimeter_se', 'area_se'
]

@app.route('/')
def home():
    return render_template('frontend.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = [float(request.form[feature]) for feature in FEATURES]
    scaled_input_data = scaler.transform([input_data])
    
    # Convert to numpy array and reshape
    input_array = np.array(scaled_input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)
    
    # Convert prediction to text
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    
    return render_template('frontend.html', 
                         prediction=result,
                         features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True)