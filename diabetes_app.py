import streamlit as st
import numpy as np
import pandas as pd 
import joblib
import os
from streamlit import sidebar
st.write("""
# MSDE4 : Projet Azure prediction diabete
## by: LAAKAD Mouad & Chrij Mohamed Amine 

""")
path=os.getcwd()
model_path=os.path.join(path,"model.pkl")
model = joblib.load(model_path)
st.sidebar.header('Veuillez insérer les parametres permettant d"évaluer Notre potentiel Client')

#######-------------------Lecture de données depuis Streamlit------------------#######
#--------------------------------------------------------------------------------------
def user_input_features():
    Pregnancies=sidebar.slider("Pregnancies",0,35,11,step=1)
    Glucose =sidebar.slider("Glucose",0,35,11,step=1)
    BloodPressure=sidebar.slider("BloodPressure",0,35,11,step=1)
    SkinThickness=sidebar.slider("SkinThickness",0,35,11,step=1)
    Insulin=sidebar.slider("Insulin",0,35,11,step=1)
    BMI=sidebar.slider("BMI",0,35,11,step=1)
    DiabetesPedigreeFunction=sidebar.slider("DiabetesPedigreeFunction",0,35,11,step=1)
    Age=sidebar.slider("Age",0,35,11,step=1)

    data={'Pregnancies': Pregnancies, 
     'Glucose': Glucose , 
     'BloodPressure': BloodPressure,
      'SkinThickness' : SkinThickness, 
      'Insulin': Insulin , 
       'BMI': BMI,
         'DiabetesPedigreeFunction': DiabetesPedigreeFunction , 
          'Age' : Age 
    }
    #features = pd.read_csv("test.csv")
    features =pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#######-------------------Prédiction et affichage du résultat ------------------#######
#--------------------------------------------------------------------------------------
st.subheader('User Input parameters')
st.write(df)
if st.sidebar.button('Predict diabete'):
  prediction = model.predict(df)
  prediction_proba = model.predict_proba(df)
  st.subheader('Class labels and their corresponding index number')
  st.write(pd.DataFrame(model.classes_))

  st.subheader('Prediction')
  dict_lend={0:"not a diabete" ,1: "diabete"}
  prediction
  prediction_str=dict_lend[prediction[0]]
  st.write(prediction_str)
  st.subheader('Prediction Probability')
  st.write(prediction_proba)
