import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from datasets import list_datasets, load_dataset

# nltk.download('punkt')


# cargar dataset de huggingface
def cargar_dataset (name_dataset):
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
def cargar_dataset_csv ():
    dataframe = pd.DataFrame()
    dataframe = pd.read_csv('./datasets/dataset_tesis.csv', encoding='ISO-8859-1', sep='|', engine='python')
    dataframe.columns = ['texto','autor_nombre','autor_apellido','titulo','año','carrera']
    print(dataframe.head())
    print(dataframe.columns)
    return dataframe


def preprocess (text):
    #text = "A las 8 en punto de la mañana ... Arthur no se sentía bien. 21 18-5-2022 yisel@gmail.com"
    text = text.lower()  # convertir a minúsculas

    # tokenizar
    tokens = nltk.word_tokenize(text,
                                language="spanish")  # no elimina signos de puntuación, separa los correos, mantiene las fechas
    # tokenizer = nltk.RegexpTokenizer(r"\w+") # elimina signos de puntuación, separa las fechas y los correos
    # tokenizer = nltk.RegexpTokenizer('\w+|\$[\d\.]+|\S+') # no elimina signos de puntuación, separa las fechas y los correos
    # tokens = tokenizer.tokenize(sentence)
    print(tokens)
    tokens = set(tokens)  # eliminar palabras repetidas
    words = [word for word in tokens if word.isalnum()]  # elimina los string que no son alfanuméricos
    print(words)

    # eliminar palabras vacías
    # print(stopwords.words('spanish')) #lista de palabras vacías
    stop_words = set(stopwords.words('spanish'))
    #clean_words = [w for w in words if not w.lower() in stop_words]
    clean_words = []
    for w in words:
        if w not in stop_words:
            clean_words.append(w)
    print(clean_words)

    # stemming: eliminar plurales
    stemmer = SnowballStemmer("spanish")
    singles_words = [stemmer.stem(plural) for plural in clean_words]
    print(singles_words)
    return singles_words


# PROCESAR EL DATASET:

# leer datset de huggingface
# train_ds = cargar_dataset('tesis')
# texto = train_ds[:'texto']
# titulo = train_ds[:'titulo']
# carrera = train_ds[:'carrera']

# leer dataset en csv
df = cargar_dataset_csv()
texto = df.loc[:, 'texto']
titulo = df.loc[:, 'titulo']
carrera = df.loc[:, 'carrera']
textos_procesados = []
titulos_procesados = []

for i in range(len(texto)):
    #print(texto[i])
    textos_procesados.append(preprocess(texto[i]))

for j in range(len(titulo)):
    #print(texto[j])
    titulos_procesados.append(preprocess(titulo[j]))

file = open('datasets/dataset_tesis_procesado.csv', 'w', encoding='ISO-8859-1')
file.write('texto|titulo|carrera' + '\n')
for k in range(len(textos_procesados)):
    for a in textos_procesados[k]:
        #print(a)
        file.write(str(a) + ' ')
    file.write('|')
    for b in titulos_procesados[k]:
        #print(b)
        file.write(str(b) + ' ')
    file.write('|' + str(carrera[k]) + '\n')
file.close()
