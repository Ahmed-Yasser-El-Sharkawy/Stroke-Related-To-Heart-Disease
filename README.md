# 🌍 Sahha: AI-Powered Stroke Risk Prediction and Monitoring System

## 🚀 Project Overview

**Sahha** is a multi-component intelligent health platform built to provide real-time cerebrovascular stroke and heart disease risk assessment. This project unites machine learning, embedded systems, computer vision, and cloud computing to offer a comprehensive healthcare solution for early diagnosis and personal monitoring.
![image](https://github.com/user-attachments/assets/f04f420c-a197-4571-a004-ab40156f1b58)

---

## 🧠 AI-Powered Risk Prediction Module

### 🧩 Model Architecture

* **Nested Structure**: First, a **heart disease prediction model** processes the user data. If risk is detected, the result is passed as an additional input to the **stroke prediction model**.
* # Models Architecture
* 
![image](https://github.com/user-attachments/assets/2c803ca3-72fe-40a9-9126-30890a3dacad)

*# Models Architecture Video Demo

[https://github.com/user-attachments/assets/f21494ee-3b67-4439-826f-e1b240da4491](https://github.com/user-attachments/assets/f21494ee-3b67-4439-826f-e1b240da4491)

* **Goal**: Improve stroke prediction accuracy by incorporating cardiovascular conditions as an influencing factor.

### 📊 Datasets
![image](https://github.com/user-attachments/assets/bcbd5102-a9f2-4fdc-b906-1d4b885e1341)

* **Stroke Dataset**: [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

  * Fields: `gender`, `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`
* **Heart Disease Dataset**: [UCI / Kaggle Heart Dataset](https://www.kaggle.com/datasets/abhishek14398/heart-disease-classification)

  * Fields: `age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`, `max_hr`, `oldpeak`, `slope`, `thal`, `target`

### ❗ Common AI Failure Scenarios and Limitations

Despite its strengths, the AI module has several failure conditions and limitations that should be noted:

#### 1. **Data Bias and Imbalance**

* Stroke and heart disease datasets are often imbalanced (fewer positive stroke cases).
* Models may develop a bias toward the majority class (healthy individuals), reducing sensitivity for high-risk patients.
* **Mitigation**: Oversampling (SMOTE), under-sampling, and class weighting strategies were implemented.

#### 2. **Feature Redundancy or Noise**

* Some features like `smoking_status` or `work_type` can introduce noise if inconsistently labeled.
* **Impact**: Reduces model interpretability and performance.
* **Solution**: Feature selection and PCA were considered to reduce dimensionality.

#### 3. **Generalization to Unseen Data**

* Overfitting may occur due to the limited size and diversity of datasets.
* Users from different ethnic, regional, or demographic backgrounds may not be well represented.
* **Approach**: K-Fold Cross Validation and real-time testing with BLE sensor data.

#### 4. **Input Errors from Embedded System**

* The wearable device may send noisy or faulty data (e.g., missing ECG signal, irregular pulse).
* **Consequence**: Can lead to inaccurate predictions.
* **Countermeasure**: Implemented sanity checks and thresholds for real-time filtering before AI processing.

#### 5. **Latency and Connectivity Issues**

* Since predictions are cloud-based (Flask API on GCP), poor connectivity can delay or interrupt predictions.
* **Solution**: Offline fallback alerts and lightweight on-device pre-checks are under development.

#### 6. **Limited Contextual Understanding**

* The AI model relies solely on structured numeric data and lacks access to patient history or doctor notes.
* **Risk**: May miss nuanced health patterns that are obvious to clinicians.
* **Future Direction**: Integration with EHR systems or NLP-based medical record parsing.

#### 7. **Interpretability Challenges**

* Machine learning models (e.g., Random Forest, XGBoost) provide limited insight into exact decision logic.
* **Enhancement**: SHAP and LIME were used to visualize feature contributions to individual predictions.

These considerations are critical when evaluating model outputs for clinical decision support. Our system is designed to assist—not replace—medical professionals.

---

### 🛠️ Technologies Used

* Libraries: `Scikit-learn`, `Pandas`, `NumPy`, `Matplotlib`
* Preprocessing: Outlier removal, missing value imputation, label encoding
* Training: Stratified splitting, hyperparameter tuning, evaluation (AUC, F1, Accuracy)

---

## 🌐 API & Deployment

### ⚙️ Flask-Based REST API

* Built using **Flask**, returns stroke risk probabilities.
* Hosted on **Google Cloud Platform** using App Engine.
* Scalable, fast, and secure backend infrastructure.

### 🔁 Data Flow

1. **User Input**: Manual form or sensor-based via mobile app
2. **Backend API**: Heart model runs → result forwarded to stroke model
3. **Output**: Single risk prediction (score) returned to front-end

---

## 🩺 Chatbot Module: Sahha LLM

### 🧠 LLM Training Pipeline

* Model: Meta-LLaMA-3.1-8B (fine-tuned)
*![image](https://github.com/user-attachments/assets/1d0b75ed-03ec-47a6-9867-792e59c0a4ae)

* Quantization: 4-bit for optimal performance on edge devices
* Source Material: Stroke-focused textbooks (cleaned & structured via GPT/Gemini)
* #Train/Loss LLM Model
* ![image](https://github.com/user-attachments/assets/4375b5ca-4d6f-425d-b766-7c89434f000a)
* Model: [Finetuned LLM](https://huggingface.co/Ahmed-El-Sharkawy/Stroke-medical-model-finetuned)

### 🧾 Use Cases

* Medical Q\&A
* Student and patient education
* Embedded assistant in app

---

## 🧠 Computer Vision Stroke Detection

### 🧬 CNN for Brain CT Scan Analysis
![image](https://github.com/user-attachments/assets/3148d1a6-302b-4aa1-8871-f4fca0bb4475)

* Model Type: Custom CNN
* Task: Binary classification (`Normal`, `Stroke`) and then (`Hemorrhagic` ,`ischemic`)
* Layers: Convolution → ReLU → MaxPooling → Fully Connected → Softmax
* ![image](https://github.com/user-attachments/assets/a62be57e-32c5-427a-a7e7-65a272525029)
* Output: Classification label + heatmap localization (green bounding box)

### 📸 Sample Workflow

1. Upload brain CT scan (224x224x3)
2. Preprocess (rescale, normalize)
![image](https://github.com/user-attachments/assets/0b9d6115-f82e-4b7f-81eb-39b02b36d6e4)
4. Predict stroke label
5. Visual output with highlighted region (via OpenCV + Matplotlib)
![image](https://github.com/user-attachments/assets/c604044e-84c3-49e4-8a90-313b44e0f986)


---

## 🧩 Embedded System
![image](https://github.com/user-attachments/assets/437fdfa2-de42-43ef-9de7-36f663b62873)

### 🛠️ Hardware Design
![image](https://github.com/user-attachments/assets/f3a30881-2480-420d-b7a3-e59797457e2e)

* **Microcontroller**: Arduino Nano 33 IoT (BLE + WiFi)
  ![image](https://github.com/user-attachments/assets/2e6457f5-4613-48c4-93c7-fc781b3d81ae)

* **Sensors**:

  * MAX30102 (HR + SpO2)
    ![image](https://github.com/user-attachments/assets/0fb64971-286a-4f74-92cf-53be061a985f)
    ![image](https://github.com/user-attachments/assets/10ae7f1d-adef-4c7d-b1aa-4b2324ebf301)
  * ECG module (AD8232)
    ![image](https://github.com/user-attachments/assets/cf8cced3-a6ca-400b-bee9-55c6442d711c)
    ![image](https://github.com/user-attachments/assets/7f7e61cb-9de9-4f21-a909-7741f0ea35b2)
* **Display**: OLED 128x64 I2C
  ![image](https://github.com/user-attachments/assets/3156f59e-1291-43f8-a27f-c93c9c13953f)
  ![image](https://github.com/user-attachments/assets/75c833bb-d7fc-4e9c-b581-62a0f18a176e)
* **User Controls**: 3 Push Buttons (Mode Switching)
 ![image](https://github.com/user-attachments/assets/af4f9991-fced-4756-b8bf-ab166f65dc8c)
* **Feedback**: Piezo buzzer (synchronized with heartbeat)
  ![image](https://github.com/user-attachments/assets/d88627a9-c69f-4482-8fea-6bbe6afbe10b)
* **Power**: USB or Li-ion battery pack (portable use) and Powered by a 350mAh rechargeable lithium battery.
![image](https://github.com/user-attachments/assets/6eb94a1f-50da-47f9-b0d0-1b5a08c6592d)
![image](https://github.com/user-attachments/assets/945cf41a-9bb4-4645-9d73-ff8c2b9780b5)

### 🔄 Workflow
![image](https://github.com/user-attachments/assets/0ca4653c-5fda-473c-8ae4-9eeda843db32)

1. User selects sensor with button
2. Data acquired & processed on Arduino
3. OLED displays result
4. BLE sends result to app
5. Flask API receives → Stroke prediction returned

### 🧮 Signal Processing

* MAX30102: Pulse waveform + digital filtering
* ECG: Analog filtering + moving average smoothing
* Display + Buzzer sync for live biofeedback

---

## 📱 Mobile Application Interface

### 🔐 Login Screen
![image](https://github.com/user-attachments/assets/9f94ffea-3172-4d5d-96b6-4c0e7d966bff)
* Firebase Auth: Email + Google sign-in

### 📊 Dashboard
![image](https://github.com/user-attachments/assets/a8ceb130-801b-4a8a-b86c-c40133392fee)

* Real-time health summary
* Recent prediction result & tips
* Chatbot access

### 👤 Profile
![image](https://github.com/user-attachments/assets/be7f913b-2b2c-40ae-8eb6-55c5d3da5ed4)

* Personal & medical details stored in Firestore
* BMI, smoking, marriage status, etc.

### 🧾 Edit Profile
![image](https://github.com/user-attachments/assets/e7106182-f38b-4b6a-ae81-1b9ef0317976)
* Sync changes with Firestore in real time

### 🔐 Change Password
![image](https://github.com/user-attachments/assets/2493dfd1-d964-465c-8fb7-d42a93c242a4)
* Firebase Auth secure password reset

### 🧪 Self-Check
![image](https://github.com/user-attachments/assets/9357f36c-6c8f-4070-883c-9c3d97af0228)

* BLE streaming of HR, SpO2, ECG
* Sent to API → Predict → Display results

### 📦 Device Request
![image](https://github.com/user-attachments/assets/7feafd77-4544-40b2-b5ca-aaa2ea71e710)

* Request hardware device
* Upload Vodafone Cash receipt
* Track order with live GPS

### 💬 Chatbot
![image](https://github.com/user-attachments/assets/b02a1825-8454-4fd5-bfb0-7c6c57ce8c53)

* Ask Sahha for medical guidance
* Natural-language interface

### ⚙️ Settings
![image](https://github.com/user-attachments/assets/a565a46a-97d9-47ff-901d-e7879cd25e0a)

* Dark mode
* Secure sign out
* Account privacy controls

---

## 🔐 Firebase Backend

### 🧩 Modules

* Firebase Auth: Identity management
  ![image](https://github.com/user-attachments/assets/4262ae00-8ba2-4ece-8d89-e05fde092cbf)

* Firestore: Health records, requests, chat logs
  ![image](https://github.com/user-attachments/assets/d969eab2-d4f6-4440-8026-91909c2a09b9)

* Firebase Storage: Upload & manage payment proofs
![image](https://github.com/user-attachments/assets/b2463a31-453d-47ff-96ca-366c6dec1b42)

---

## ☁️ Cloud Infrastructure

### ⚙️ GCP Features

* Flask API deployed via Google App Engine
* 99.9% uptime with autoscaling
* Real-time predictions from ML models
* Fast & scalable for growing users

---

## 🧪 Technologies Overview

| Component       | Stack / Platform             |
| --------------- | ---------------------------- |
| AI Prediction   | Python, Scikit-learn, Pandas |
| Computer Vision | TensorFlow, OpenCV, Keras    |
| Chatbot LLM     | HuggingFace, Transformers    |
| Backend         | Flask, Google App Engine     |
| Database        | Firebase Firestore, Storage  |
| App             | Flutter (Dart)               |
| Hardware        | Arduino Nano 33 IoT          |
| Communication   | Bluetooth Low Energy (BLE)   |

---


## 🏆 Key Achievements

* End-to-end AI & IoT health monitoring platform
* Stroke risk predicted with heart condition influence
* Integrated chatbot assistant trained on expert data
* Real-time BLE communication from wearable to app

---

## 🗺️ Roadmap

* Add blood pressure sensing
* Expand chatbot to cover diabetes, hypertension
* Multi-language UI (Arabic, French, English)

---

## 👨‍💻 Contributors

| Name                                                                    | Role                                                             | GitHub                        |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------- |
| ![AhmedAmr](https://avatars.githubusercontent.com/u/162978338?s=64)     | [Ahmed Amr](https://github.com/Ahmedamr778)                      | **Mobile Developer**          |
| ![AhmedAli](https://avatars.githubusercontent.com/u/159345376?s=64)     | [Ahmed Ali Abd-Elshafy](https://github.com/Ahmed209Ali)          | **Embedded Systems Engineer** |
| ![Ahmed](![image](https://github.com/user-attachments/assets/7f1a1732-da74-4816-911e-2b1c20d7873f))  | [Ahmed El-Sharkawy](https://github.com/Ahmed-Yasser-El-Sharkawy) | **AI Team Lead**              |
| ![AhmedYoussef](https://avatars.githubusercontent.com/u/163133239?s=64) | [Ahmed Mohamed Youssef](https://github.com/AhmedGad231)          | **AI Developer**              |
| ![Ibrahim](https://avatars.githubusercontent.com/u/163135576?s=64)      | [Ibrahim Ehab](https://github.com/ibrahimehab0222)               | **AI Developer**              |

 
> 🔗 Visit each profile to explore more contributions and repositories!

---

## 📜 License

MIT License © 2025

> Health is not a privilege—it's a right. Sahha empowers individuals with the knowledge and tools to protect it.
