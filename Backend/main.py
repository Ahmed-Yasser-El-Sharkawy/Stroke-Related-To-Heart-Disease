from flask import Flask, request, jsonify
from evel import PersonData
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

# --- Models ---
class EnhancedCNN_CT(nn.Module):
    def __init__(self):
        super(EnhancedCNN_CT, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class Sub_Class_CNNModel_CT(nn.Module):
    def __init__(self, num_classes=2):
        super(Sub_Class_CNNModel_CT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)

def preprocess_ct(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_cv, (224, 224))
    img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img_pil).unsqueeze(0)

def preprocess_sub_ct(img):
    img = img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- Inference Functions ---
def classify_ct(image):
    model = EnhancedCNN_CT()
    model.load_state_dict(torch.load('CT/best_model_CT.pth', map_location='cpu'))
    model.eval()
    tensor = preprocess_ct(image)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.sigmoid(output).item()

    if pred < 0.5:
        return ("Normal", 1 - float(pred))

    sub_model = Sub_Class_CNNModel_CT()
    sub_model.load_state_dict(torch.load('CT/cnn_model_sub_class.pth', map_location='cpu'))
    sub_model.eval()
    tensor_sub = preprocess_sub_ct(image)
    with torch.no_grad():
        sub_output = sub_model(tensor_sub)
        sub_pred = torch.argmax(sub_output, dim=1).item()
        sub_conf = sub_output[0][sub_pred].item()

    sub_class_names = ['hemorrhagic', 'ischaemic']
    return (f"Stroke - {sub_class_names[sub_pred]}", float(sub_conf))

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

@app.route('/predict_computer_vision', methods=['POST'])
def predict_computer_vision():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image = Image.open(file.stream)

        result, confidence = classify_ct(image)

        return jsonify({
            'main_prediction': result,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
