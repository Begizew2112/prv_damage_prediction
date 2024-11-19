import joblib
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Path to your model file
model_filename = r'C:\Users\Yibabe\Desktop\prv_damage_prediction\notebook\pressure_regulating_valve_model.joblib'

# Load the trained model
model = joblib.load(model_filename)

@app.route('/')
def home():
    return "Pressure Regulating Valve Damage Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Prepare data (you may need to adjust based on your model's input)
        features = [
            data['Pressure Input (kPa)'],
            data['Pressure Output (kPa)'],
            data['Pressure Difference (kPa)'],
            data['Flow Rate (m³/s)'],
            data['Temperature (°C)'],
            data['Operating Time (Hours)'],
            data['Valve Size (mm)'],
            data['Maintenance Frequency (Months)'],
            data['Failure History'],
            data['Load Cycles'],
            data['Material Type_Brass'],
            data['Material Type_Polymer'],
            data['Material Type_Steel'],
            data['Environmental Conditions_Corrosive'],
            data['Environmental Conditions_Humid'],
            data['Environmental Conditions_Normal']
        ]

        # Predict using the model
        prediction = model.predict([features])
        prediction_proba = model.predict_proba([features])[0][1]  # Probability of failure

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_of_failure': prediction_proba
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
