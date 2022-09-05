# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


data_sample = pd.DataFrame({"Pregnancies": pd.Series([0], dtype="int8"), "Glucose": pd.Series([0], dtype="int16"), "BloodPressure": pd.Series([0], dtype="int8"), "SkinThickness": pd.Series([0], dtype="int8"), "Insulin": pd.Series([0], dtype="int16"), "BMI": pd.Series([0.0], dtype="float32"), "DiabetesPedigreeFunction": pd.Series([0.0], dtype="float32"), "Age": pd.Series([0], dtype="int8")})
print(data_sample.values)

path =os.getcwd()
model_path=os.path.join(path,"model.pkl")
model=pickle.load(open(model_path,'rb'))
model.predict(data_sample.values[0])




#'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'

