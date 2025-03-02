import pandas as pd
import numpy as np
from Deploy.data import StrokeData,HeartData
from Deploy.evel import HealthPredictor,PersonData
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import joblib
import pickle  

import warnings
warnings.filterwarnings("ignore")

person_data = PersonData(63, 1, 3, 145, 0, 150, 0, 2.3, 0, 1, 0, "Yes", "Private", 228.69, 36.6, "formerly smoked")
print("Stroke Prediction:", person_data.stroke_prediction)
