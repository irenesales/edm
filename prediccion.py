import streamlit as st
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import urllib.request
import requests
import zipfile
import io


url = 'https://github.com/irenesales/edm/raw/main/valenbisi_procesado_coordenadas.zip'
response = requests.get(url)

# Leer el contenido del archivo comprimido en un objeto ZipFile
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extraer el nombre del archivo CSV dentro del archivo comprimido
csv_file_name = zip_file.namelist()[0]

# Leer el archivo CSV dentro del archivo comprimido y cargarlo en un DataFrame
data = pd.read_csv(zip_file.open(csv_file_name))


#CREAR MODELO
# Seleccionar las variables relevantes para la predicción
variables = ['Dia', 'Mes', 'Año', 'Hora','name']
target = 'avg_av'

# Realizar codificación one-hot para la variable "name"
name_encoder = OneHotEncoder(sparse=False)
column_transformer = make_column_transformer((name_encoder, ['name']), remainder='passthrough')
data_encoded = column_transformer.fit_transform(data[variables])

# Obtener los nombres de las características después de la codificación one-hot
name_categories = list(column_transformer.named_transformers_['onehotencoder'].categories_[0])
feature_names = name_categories + variables[1:]

# Crear el conjunto de datos de entrenamiento
X = data_encoded
y = data[target]

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X, y)


st.title("Predicción de disponibilidad")



 

