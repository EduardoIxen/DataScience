import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import util.CargarArchivo as cargar

st.title("Clasificador Gaussiano")

archivo = st.file_uploader("Seleccione un archivo", type=['csv', 'json', 'xml', 'xlsx', 'xls'])

if archivo is not None:
    datos = cargar.cargar_archivo(archivo)
    st.write("## Contenido del archivo")
    st.dataframe(datos)

    input_feature = st.text_input("Ingrese una lista de predicciones separadas por coma",help="1,5,8,3,7")
    input_array = input_feature.split(",")

    parameter_encoder = st.checkbox("Implementar encoder")
    parameter_tareget = st.text_input("Ingrese en nombre de la columna a codificar")

    if st.button("Calcular"):
        try:
            if parameter_encoder:
                list_col = datos.drop(columns=parameter_tareget)
                le = LabelEncoder()
                for col_name in list_col:
                    datos[col_name] = le.fit_transform(datos[col_name])

                datos[parameter_tareget] = le.fit_transform(datos[parameter_tareget])
                st.write("Nueva tabla encoder")
                st.write(datos)

            y = datos[[parameter_tareget]] #validad parametros de localizacion en la tabla sean los correctos
            x = datos.drop(columns=parameter_tareget)

            #entrenamiento y prueba de los campos
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None, random_state=None)

            #Modelo a utilizar y poner a prueba la prediccion
            model = GaussianNB()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            #array con los datos a predecir
            desired_array = [int(numeric_string) for numeric_string in input_array]
            x_new_val = np.array(desired_array)
            y_pred = model.predict([x_new_val])
            st.write("Prediccion", y_pred)
        except:
            st.info("Parametros no reconocidos")