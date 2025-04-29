from flask import Flask, request, jsonify
from evel import PersonData
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "✅ Sahha Health Prediction API is Running", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        person_data = PersonData(
            age=data['age'],
            sex=data['sex'],
            chest_pain_type=data['chest_pain_type'],
            resting_bp=data['resting_bp'],
            restecg=data['restecg'],
            max_hr=data['max_hr'],
            exang=data['exang'],
            oldpeak=data['oldpeak'],
            slope=data['slope'],
            thal=data['thal'],
            hypertension=data['hypertension'],
            ever_married=data['ever_married'],
            work_type=data['work_type'],
            avg_glucose_level=data['avg_glucose_level'],
            bmi=data['bmi'],
            smoking_status=data['smoking_status']
        )

        return jsonify({
            'heart_prediction': int(person_data.heart_prediction),
            'stroke_prediction': int(person_data.stroke_prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
