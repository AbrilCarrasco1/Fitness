import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de Fitness ''')
st.image("salud.jpg", caption="Predicción de Fitness.")

st.header('Datos')

def user_input_features():
  # Entrada
  age = st.number_input('age:', min_value=0, max_value=100, value = 0, step = 1)
  height_cm = st.number_input('height_cm:',  min_value=0.0, max_value=300.0, value = 0.0, step = 1.0)
  weight_kg = st.number_input('weight_kg:', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  heart_rate  = st.number_input('heart_rate:', min_value=0.0, max_value=1000000.0, value = 0.0, step = 1.0)
  blood_pressure  = st.number_input('blood_pressure:', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  sleep_hours  = st.number_input('sleep_hours:', min_value=0, max_value=24, value = 0, step = 1)
  nutrition_quality  = st.number_input('nutrition_quality:', min_value=0, max_value=10, value = 0, step = 1)
  activity_index  = st.number_input('activity_index:', min_value=0, max_value=5, value = 0, step = 1)
  smokes  = st.number_input('smokes:', min_value=0, max_value=1, value = 0, step = 1)
  gender  = st.number_input('gender:', min_value=0, max_value=1, value = 0, step = 1)



  user_input_data = {'age': age,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'sleep_hours': sleep_hours,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,
                     'smokes': smokes,
                     'gender': gender,
                    }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('Fitness_Classification2_df.csv', encoding='latin-1')

# Define the features to use for training the model, matching the user input features
feature_columns = ['age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure',
                   'sleep_hours', 'nutrition_quality', 'activity_index', 'smokes', 'gender']
X = datos[feature_columns] # Changed this line
y = datos['is_fit']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614954)
LR = LinearRegression()
LR.fit(X_train,y_train)

# Removed b1 and b0 assignments as they are no longer explicitly used
# b1 = LR.coef_
# b0 = LR.intercept_
prediccion = LR.predict(df) # Changed this line to use the model's predict method

st.subheader('Cálculo de fitness')
st.write('Usted es 1 fitness 0 no fitness: ', prediccion)
