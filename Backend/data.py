class HeartData:
    def __init__(self, age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal):
        self.features = [age, sex, chest_pain_type, resting_bp, restecg, max_hr, exang, oldpeak, slope, thal]

class StrokeData:
    def __init__(self, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status):
        self.features = [age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status]