import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import util.CargarArchivo as cargar

st.title("Regresión Lineal")

archivo = st.file_uploader("**Seleccione el archivo de datos**", type=['csv', 'json', 'xlsx', 'xls', 'xml'])
if archivo is not None:
    datos = cargar.cargar_archivo(archivo)
    st.write("### Contenido del archivo")
    st.dataframe(datos)

    columna1, columna2 = st.columns(2)
    with columna1: #usar with para insertar cualquier elemento en la columna
        st.write("#### Variable independiente X :")
        variableX = st.selectbox("Seleccione la variable **independiente**", datos.keys())

    with columna2:
        st.write("#### Variable dependiente Y : ")
        variableY = st.selectbox("Seleccione la variable **dependiente**", datos.keys())

    st.write("### PREDICCION")
    prediccion = st.number_input("Valor a predecir")

    #analizando los datos
    puntosX = np.asarray(datos[variableX]).reshape(-1, 1)
    puntosY = datos[variableY]

    #regresion lineal
    regre = LinearRegression()
    regre.fit(puntosX, puntosY)
    predic = regre.predict(puntosX)
    r2 = r2_score(puntosY, predic)
    valorPredic = regre.predict([[prediccion]])

    #Graficar
    grafica = plt.figure()
    plt.style.use("bmh")
    plt.scatter(puntosX, puntosY, color="red")
    plt.plot(puntosX, predic, color="blue")
    plt.title("Grafica de puntos")
    plt.xlabel(variableX)
    plt.ylabel(variableY)

    #mostrar imagen
    if st.button('Analizar datos'):
        st.write("## Gráfica")
        st.pyplot(grafica)

        pendiente = round(float(regre.coef_), 3)
        interseccion = round(float(regre.intercept_), 3)
        st.write("## Función de tendencia")
        st.latex(f"f(x) = {pendiente}x {'+' if interseccion >= 0 else ''} {interseccion}")

        st.write("## Información de la grafica")
        columna1, columna2, columna3 = st.columns(3)

        columna1.metric("Pendiente", pendiente, "-"if pendiente < 0 else "+")
        columna2.metric("Interseccion", interseccion, "-" if interseccion < 0 else "+")
        columna3.metric("R²", round(r2, 4))
        st.metric("Error R²", mean_squared_error(puntosY, predic))

        st.subheader("Predicción")
        st.metric(f"Para **{prediccion}** el resultado es: ", round(float(valorPredic), 3), "-" if valorPredic < 0 else "+")
