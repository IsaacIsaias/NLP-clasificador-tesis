import re
import string
from datetime import date

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from datasets import list_datasets, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)

# nltk.download('punkt')


# cargar dataset de huggingface
def load_dataset_hf(name_dataset):
    ds = load_dataset(name_dataset)
    print(ds)
    train_ds = ds["train"]
    print(train_ds)
    print(len(train_ds))
    print(train_ds[0])
    print(train_ds.column_names)
    print(print(train_ds.features))
    print(print(train_ds[:5]))
    return train_ds


# cargar dataset en formato csv
def load_dataset_csv():
    dataframe = pd.read_csv('./datasets/dataset_tesis.csv', encoding='utf-8', sep='|', engine='python')
    print(dataframe.head())
    dataframe.columns = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']
    print(dataframe.head())
    df = dataframe.groupby(["carrera"])["texto"].count()
    print(df)
    return dataframe


def load_dataset_csv_procesado():
    dataframe = pd.read_csv('./datasets/dataset_tesis_procesado.csv', encoding='ISO-8859-15', sep='|', engine='python')
    print(dataframe.head())
    dataframe.columns = ['texto', 'titulo', 'carrera']
    print(dataframe.head())
    df = dataframe.groupby(["carrera"])["texto"].count()
    print(df)
    return dataframe


def preprocess(text):
    # text = "A las 8 en punto de la mañana ... Arthur no se sentía bien. 21 18-5-2022 yisel@gmail.com"
    text = text.lower()  # convertir a minúsculas

    # tokenizar
    tokens = nltk.word_tokenize(text,
                                language="spanish")  # no elimina signos de puntuación, separa los correos, mantiene las fechas
    # tokenizer = nltk.RegexpTokenizer(r"\w+") # elimina signos de puntuación, separa las fechas y los correos
    # tokenizer = nltk.RegexpTokenizer('\w+|\$[\d\.]+|\S+') # no elimina signos de puntuación, separa las fechas y los correos
    # tokens = tokenizer.tokenize(sentence)
    #print(tokens)
    # tokens = set(tokens)  # eliminar palabras repetidas pero no quedan en orden de aparición
    words = [word for word in tokens if word.isalnum()]  # elimina los string que no son alfanuméricos
    #print(words)

    # eliminar palabras vacías
    # print(stopwords.words('spanish')) #lista de palabras vacías
    stop_words = set(stopwords.words('spanish'))
    # clean_words = [w for w in words if not w.lower() in stop_words]
    clean_words = []
    for w in words:
        if w not in stop_words:
            clean_words.append(w)
    #print(clean_words)

    # stemming: eliminar plurales
    stemmer = SnowballStemmer("spanish")
    singles_words = [stemmer.stem(plural) for plural in clean_words]
    print(singles_words)
    return singles_words


# PROCESAR EL DATASET:
def process_all_dataset():
    # leer datset de huggingface
    # train_ds = cargar_dataset('tesis')
    # texto = train_ds[:'texto']
    # titulo = train_ds[:'titulo']
    # carrera = train_ds[:'carrera']

    # leer dataset en csv
    df = load_dataset_csv()
    texto = df.loc[:, 'texto']
    titulo = df.loc[:, 'titulo']
    carrera = df.loc[:, 'carrera']
    textos_procesados = []
    titulos_procesados = []

    for i in range(len(texto)):
        print(texto[i])
        textos_procesados.append(preprocess(texto[i]))

    for j in range(len(titulo)):
        print(titulo[j])
        titulos_procesados.append(preprocess(titulo[j]))

    file = open('datasets/dataset_tesis_procesado.csv', 'w', encoding='ISO-8859-15', errors='ignore')  # ISO-8859-1
    file.write('texto|titulo|carrera' + '\n')
    for k in range(len(textos_procesados)):
        for a in textos_procesados[k]:
            file.write(str(a) + ' ')
        file.write('|')
        for b in titulos_procesados[k]:
            file.write(str(b) + ' ')
        file.write('|' + str(carrera[k]) + '\n')
    file.close()


def split_dataset():
    df = load_dataset_csv()
    x = df.loc[:, df.columns != 'carrera']
    y = df.loc[:, 'carrera']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train = pd.DataFrame()
    #train.columns = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']
    train['texto'] = x_train ['texto']
    train['autor_nombre'] = x_train['autor_nombre']
    train['autor_apellido'] = x_train['autor_apellido']
    train['titulo'] = x_train['titulo']
    train['año'] = x_train['año']
    train['carrera'] = y_train
    print(train)
    test = pd.DataFrame()
    # train.columns = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']
    test['texto'] = x_test['texto']
    test['autor_nombre'] = x_test['autor_nombre']
    test['autor_apellido'] = x_test['autor_apellido']
    test['titulo'] = x_test['titulo']
    test['año'] = x_test['año']
    test['carrera'] = y_test
    print(test)

    train.to_csv('datasets/dataset_tesis_train.csv', sep='|')
    test.to_csv('datasets/dataset_tesis_test.csv', sep='|')
    # file = open('datasets/dataset_tesis_train.csv', 'w', encoding='utf-8')
    # file.write('texto|autor_nombre|autor_apellido|titulo|año|carrera' + '\n')
    # for i in range(len(x_train)):
    #     file.write(str(x_train.iloc[i:]) + '|' + str(y_train[i:]) + '\n')
    # file.close()
    # file = open('datasets/dataset_tesis_test.csv', 'w', encoding='utf-8')
    # file.write('texto|autor_nombre|autor_apellido|titulo|año|carrera' + '\n')
    # for j in range(len(x_test)):
    #     file.write(str(x_test[j]) + '|' + str(y_test[j]) + '\n')
    # file.close()


def split_dataset_procesado():
    df = load_dataset_csv_procesado()
    x = df.loc[:, df.columns != 'carrera']
    y = df.loc[:, 'carrera']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train = pd.DataFrame()
    #train.columns = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']
    train['texto'] = x_train ['texto']
    train['titulo'] = x_train['titulo']
    train['carrera'] = y_train
    print(train)
    test = pd.DataFrame()
    # train.columns = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']
    test['texto'] = x_test['texto']
    test['titulo'] = x_test['titulo']
    test['carrera'] = y_test
    print(test)

    train.to_csv('datasets/dataset_tesis_procesado_train.csv', sep='|')
    test.to_csv('datasets/dataset_tesis_procesado_test.csv', sep='|')


process_all_dataset()
split_dataset()
split_dataset_procesado()
#load_dataset_csv()
# from datasets import load_dataset
# dataset = load_dataset("hackathon-pln-es/unam_tesis", split='train')
# print(dataset)
