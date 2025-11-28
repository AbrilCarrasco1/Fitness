import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de Fitness ''')
st.image("salud.jpg", caption="Predicción de Fitness.")

st.header('Datos')

def user_input_features():
  # Entrada
  age = st.number_input('Edad:', min_value=0, max_value=100, value = 0, step = 1)
  height_cm = st.number_input('Estatura (cm):',  min_value=0.0, max_value=300.0, value = 0.0, step = 1.0)
  weight_kg = st.number_input('Peso (Kg):', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  heart_rate  = st.number_input('Frecuencia Cardiaca:', min_value=0.0, max_value=1000000.0, value = 0.0, step = 1.0)
  blood_pressure  = st.number_input('Presión Arterial:', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  sleep_hours  = st.number_input('Horas de sueño:', min_value=0, max_value=24, value = 0, step = 1)
  nutrition_quality  = st.number_input('Calidad de alimentación (1 a 10):', min_value=0, max_value=10, value = 0, step = 1)
  activity_index  = st.number_input('Indice de actividad (1 a 5):', min_value=0, max_value=5, value = 0, step = 1)
  smokes  = st.number_input('Fumas:', min_value=0, max_value=1, value = 0, step = 1)
  gender  = st.number_input('Género:', min_value=0, max_value=1, value = 0, step = 1)



  user_input_data = {'Edad:': age,
                     'Estatura (cm):': height_cm,
                     'Peso (Kg):': weight_kg,
                     'Frecuencia Cardiaca:': heart_rate,
                     'Presión Arterial:': blood_pressure,
                     'Horas de sueño:': sleep_hours,
                     'Calidad de alimentación (1 a 10):': nutrition_quality,
                     'Indice de actividad (1 a 5):': activity_index,
                     'Fumas:': smokes,
                     'Género:': gender,
                    }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fitness =  pd.read_csv('Fitness_Classification2.csv', encoding='latin-1')
X = fitness.drop(columns='is_fit')
Y = fitness['is_fit']

classifier = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=25, max_features=5, random_state=1614954)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No Fit')
elif prediction == 1:
  st.write('Fit')
else:
  st.write('Sin predicción')
