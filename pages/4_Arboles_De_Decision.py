import streamlit as st
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import util.CargarArchivo as cargar

st.title("Árboles de decisión")

archivo = st.file_uploader("Seleccione un archivo", type=['csv', 'json', 'xml', 'xlsx', 'xls'])

if archivo is not None:
    datos = cargar.cargar_archivo(archivo)

    st.write("Contenido del archvio")
    st.dataframe(datos)

    st.write("### Parametrización")

    clasificador = st.selectbox("Variable **objetivo**", datos.keys())

    niveles = st.number_input("Niveles del arbol", None, None, 0, 1)
    niveles = None if niveles == 0 else niveles

    tamLetra = st.number_input("Tamaño de la fuente", None, None, 0, 1)
    tamLetra = None if tamLetra == 0 else tamLetra

    if st.button("Mostrar Árbol"):
        objetivo = datos[clasificador]                    #valorY
        explicativas = datos.drop([clasificador], axis=1)      #datos

        valorX = []
        label_encoder = preprocessing.LabelEncoder()
        labels = explicativas.head()
        valores = labels.columns

        for val in valores:
            lista_valores = list(explicativas[val])
            transformacion = label_encoder.fit_transform(lista_valores)
            valorX.append(transformacion)

        features = list(zip(*valorX))
        label = label_encoder.fit_transform(objetivo)

        modelo = DecisionTreeClassifier(max_depth=niveles)
        modelo.fit(X=features, y=label)    #explicativas, objetivo

        figura = plt.figure()
        plt.style.use("bmh")
        plot_tree(modelo, feature_names=explicativas.columns, class_names=objetivo, filled=True, fontsize=tamLetra)

        st.write("Todas las flechas del lado izquierdo son respuestas **Verdaderas** y las del lado derecho son respuestas **Falsas**")
        st.write("**gini** = La pureza de los nodos, mientras menor sea se tiene una mejor division de las clases.")
        st.write("**samples** = La cantidad de individuos que cumplen con esas condiciones.")
        st.write("**value** = [Individuos que SI, individuos que NO]")
        st.write("**class** = La clase de la mayoria de individuos en el nodo.")


        plt.title("Arbol de decision")
        st.pyplot(figura)
