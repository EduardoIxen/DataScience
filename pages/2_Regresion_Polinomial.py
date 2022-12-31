import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import util.CargarArchivo as cargar

st.title("Regresión Polinomial")

archivo = st.file_uploader("Seleccione un archivo", type=['csv', 'json', 'xml', 'xlsx', 'xls'])

if archivo is not None:
    datos = cargar.cargar_archivo(archivo)
    st.write("## Contenido del archivo")
    st.dataframe(datos)

    st.write("## Parametrización")

    columna1, columna2 = st.columns(2)

    with columna1: #usar with para insertar cualquier elemento en la columna
        st.write("Variable independiente X")
        variableX = st.selectbox("Seleccione la variable independiente", datos.keys())

    with columna2:
        st.write("Variable dependiente Y")
        variableY = st.selectbox("Seleccione la variable dependiente", datos.keys())

    st.write("Grado de la función")
    grado = st.radio("Seleccione el grado del modelo", (2,3,4,5), horizontal=True)

    st.write("## Predicción")
    prediccion = st.number_input("Valor a predecir", None, None, 0, 1)

    #Comenzando a analizar la informacion
    puntosX = np.asarray(datos[variableX]).reshape(-1, 1)
    puntosY = datos[variableY]

    #Configuracion de la regresion polinomial
    polinomial = PolynomialFeatures(degree=grado)
    transformacion = polinomial.fit_transform(puntosX)

    #Configuracion de la regresión lineal
    regresion = LinearRegression()
    regresion.fit(transformacion, puntosY)
    predecir = regresion.predict(transformacion)
    r2 = r2_score(puntosY, predecir)

    #Predicción
    minX = prediccion
    maxX = prediccion
    nuevaX = np.linspace(minX, maxX, 1)[:, np.newaxis]
    transformacion = polinomial.fit_transform(nuevaX)
    valorPredecido = regresion.predict(transformacion)

    #Dibujar
    figura = plt.figure()
    plt.style.use("bmh")
    plt.scatter(puntosX, puntosY, color="red")
    plt.plot(puntosX, predecir, color="blue")
    plt.title(f"Función Polinomial de grado = {grado}")
    plt.ylabel(variableY)
    plt.xlabel(variableX)

    #Mostrando la imagen
    if st.button("Analizar entrada"):
        cero = round(regresion.intercept_, 4)

        st.write("### Función de tendencia")
        contador = grado
        funcion = "f(x) = "
        while contador > 0:
            funcion += f"{'' if regresion.coef_[contador] < 0 else '+'}{round(float(regresion.coef_[contador]),8)}x{f'^{contador}' if contador != 1 else ''}"
            contador -= 1

        funcion += f"{'' if cero < 0 else '+'}{cero}"
        st.latex(funcion)

        st.write("### Grafica")
        st.pyplot(figura)

        st.write("### Información de la gráfica")
        columna1, columna2 = st.columns(2)

        with columna1:
            st.write("Coeficiente de la función")
            contador = 1
            for i in regresion.coef_[1:]:
                st.write(f"- **x^{contador}:** {i}")
                contador += 1

        with columna2:
            columna2.metric("R²", round(r2, 4))

        st.metric("Intersección", cero, "-" if cero < 0 else "+")
        st.metric("Error de R²", mean_squared_error(puntosY, predecir))

        st.subheader("Predicción")
        st.metric(f"La prediccipon para {prediccion} es: ", valorPredecido, "-" if valorPredecido < 0 else "+")


