from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import joblib
import pickle  


class HeartData:
    def __init__(self, age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal):
        self.features = [age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal]

class StrokeData:
    def __init__(self, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status):
        self.features = [age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status]

class HealthPredictor:
    def __init__(self):
        self.heart_model_path = 'Heart_Disease/Saved_Model_Status/HeartModelRandomForest'
        self.heart_scaler_path = 'Heart_Disease/Saved_Model_Status/Standard_scaler.pkl'
        self.stroke_model_path = 'Stroke_Code/Saved_Model_Status/StrokeModelRandomForest'
        self.stroke_scaler_path = 'Stroke_Code/Saved_Model_Status/Standard_scaler.pkl'
        self.encoders_paths = {
            'ever_married': 'Stroke_Code/Saved_Model_Status/ever_married_encoder.pkl',
            'work_type': 'Stroke_Code/Saved_Model_Status/work_type_encoder.pkl',
            'smoking_status': 'Stroke_Code/Saved_Model_Status/smoking_status_encoder.pkl'
        }

        self.heart_model = self.load_model(self.heart_model_path)
        self.heart_scaler = self.load_scaler(self.heart_scaler_path)
        self.stroke_model = self.load_model(self.stroke_model_path)
        self.stroke_scaler = self.load_scaler(self.stroke_scaler_path)

        self.ever_married_encoder = self.load_encoder(self.encoders_paths['ever_married'])
        self.work_type_encoder = self.load_encoder(self.encoders_paths['work_type'])
        self.smoking_status_encoder = self.load_encoder(self.encoders_paths['smoking_status'])

    def load_model(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def load_scaler(self, path):
        return joblib.load(path)

    def load_encoder(self, path):
        return joblib.load(path)

    def predict_heart(self, data_point):
        data_point_scaled = self.heart_scaler.transform(np.array([data_point]))
        return self.heart_model.predict(data_point_scaled)[0]

    def predict_stroke(self, data_point):
        data_point[3] = self.ever_married_encoder.transform([data_point[3]])[0]
        data_point[4] = self.work_type_encoder.transform([data_point[4]])[0]
        data_point[7] = self.smoking_status_encoder.transform([data_point[7]])[0]

        data_point_scaled = self.stroke_scaler.transform(np.array([data_point]))
        
        # Get prediction probability
        probabilities = self.stroke_model.predict_proba(data_point_scaled)[0]

        # You can return both prediction and probabilities if needed
        prediction = np.argmax(probabilities)
        return prediction, probabilities
        
        # return self.stroke_model.predict(data_point_scaled)[0]


class PersonData:
    def __init__(self, age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal,
                hypertension, ever_married, work_type, avg_glucose_level, bmi, smoking_status):
        self.features = [age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal,
                hypertension, ever_married, work_type, avg_glucose_level, bmi, smoking_status]
        self.predictor = HealthPredictor()
        self.heart_prediction = self.predict_heart()
        self.stroke_prediction, self.stroke_proba = self.predict_stroke()

    def predict_heart(self):
        heart_data = HeartData(*self.features[:10])
        return self.predictor.predict_heart(heart_data.features)

    def predict_stroke(self):
        self.heart_prediction
        stroke_data = StrokeData(self.features[0], self.features[10], self.heart_prediction, self.features[11], self.features[12],
                                 self.features[13], self.features[14], self.features[15])
        return self.predictor.predict_stroke(stroke_data.features)
    
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
            'stroke_prediction': int(person_data.stroke_prediction),
            'stroke_probability': round(float(person_data.stroke_proba[person_data.stroke_prediction]), 4)           
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8087, debug=True)

