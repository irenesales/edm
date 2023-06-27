import streamlit as st
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


ruta_archivo = "C:/Users/Usuario/Desktop/uni/tercero/EDM/proyecto/valenbisi_procesado_coordenadas.csv"

# Cargar el archivo CSV en un DataFrame
data = pd.read_csv(ruta_archivo)

#CREAR MODELO
# Seleccionar las variables relevantes para la predicción
variables = ['Dia', 'Mes', 'Año', 'Hora','name']
target = ['avg_av','avg_free', 'avg_total']

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



Hora = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13,14,15,16,17,18,19,20,21,22,23]
Dia = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
Mes = [1,2,3,4,5,6,7,8,9,10,11,12]
Nombre=list(data['name'].unique())
Anyo = [2023,2024]
graficos = ['Bicicletas disponibles', 'Espacios libres']

with st.form('entry_form', clear_on_submit = False):
    col1, col2, col3 = st.columns(3)
    col1.selectbox("Selecciona día:" , Dia, key="day")
    col2.selectbox("Selecciona mes:" , Mes, key="month")
    col3.selectbox("Selecciona año:" , Anyo, key="year")
    col1.selectbox("Selecciona hora:" , Hora, key="hour")
    col2.selectbox("¿Qué quieres predecir?", graficos, key='figure')
    col3.selectbox('Selecciona estación: ', Nombre, key = 'name')
    
    "---"
    submitted = st.form_submit_button(label='Guardar datos')
    if submitted:
        selected_day = st.session_state.day
        selected_month = st.session_state.month
        selected_year = st.session_state.year
        selected_hour = st.session_state.hour
        selected_name = st.session_state.name
        selected_figure = st.session_state.figure
        

        st.success('Datos guardados')


        
        if selected_figure ==  'Bicicletas disponibles':
            nuevos_datos = pd.DataFrame([[selected_day, selected_month, selected_year, selected_hour, selected_name]], columns=variables)
            nuevos_datos_encoded = column_transformer.transform(nuevos_datos)
            
            # Realizar la predicción
            prediccion = model.predict(nuevos_datos_encoded)
            
            # Mostrar la predicción
            st.subheader("Resultados de la predicción:")

            #avg_av
            st.write("Media de bicicletas disponibles: ", round(prediccion[0][0]))

            
            

        if selected_figure == 'Espacios libres':
            nuevos_datos = pd.DataFrame([[selected_day, selected_month, selected_year, selected_hour, selected_name]], columns=variables)
            nuevos_datos_encoded = column_transformer.transform(nuevos_datos)
            
            # Realizar la predicción
            prediccion = model.predict(nuevos_datos_encoded)
            
            # Mostrar la predicción
            st.subheader("Resultados de la predicción:")

            #avg_av
            st.write("Media de espacios libres: ",round(prediccion[0][1]))


            
            


 
