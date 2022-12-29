from click import File
import pandas as pd
import os


def cargar_archivo(archivo: File) -> pd.DataFrame:
    nombre = os.path.splitext(archivo.name)
    extension = nombre[1]

    if extension == '.csv':
        datos = pd.read_csv(archivo)
    elif extension == '.xls' or extension == '.xlsx':
        datos = pd.read_excel(archivo)
    elif extension == '.xml':
        datos = pd.read_xml(archivo)
    else:
        datos = pd.read_json(archivo)

    return datos
