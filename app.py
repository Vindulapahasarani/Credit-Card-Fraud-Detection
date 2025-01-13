from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array for prediction
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Map prediction to result
        result = "Fraudulent Transaction" if prediction[0] == 1 else "Normal Transaction"
        
        return render_template('index.html', prediction_text=f'Transaction Status: {result}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
