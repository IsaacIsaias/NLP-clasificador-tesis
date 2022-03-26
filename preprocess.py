import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from datasets import list_datasets, load_dataset
#nltk.download('punkt')

#cargar dataset de huggingface
# ds = load_dataset("tesis")
# print(ds)
# train_ds = ds["train"]
# print(train_ds)
# print(len(train_ds))
# print(train_ds[0])
# print(train_ds.column_names)
# print(print(train_ds.features))
#print(print(train_ds[:5]))


text = "A las 8 en punto de la mañana ... Arthur no se sentía bien. 21 18-5-2022 yisel@gmail.com"
text = text.lower() #convertir a minúsculas

# tokenizar
tokens = nltk.word_tokenize(text, language="spanish") #no elimina signos de puntuación, separa los correos, mantiene las fechas
#tokenizer = nltk.RegexpTokenizer(r"\w+") # elimina signos de puntuación, separa las fechas y los correos
#tokenizer = nltk.RegexpTokenizer('\w+|\$[\d\.]+|\S+') # no elimina signos de puntuación, separa las fechas y los correos
#tokens = tokenizer.tokenize(sentence)
print(tokens)
tokens = set(tokens) #eliminar palabras repetidas
words = [word for word in tokens if word.isalnum()] #elimina los string que no son alfanuméricos
print(words)

#eliminar palabras vacías
#print(stopwords.words('spanish')) #lista de palabras vacías
stop_words = set(stopwords.words('spanish'))
filtered_sentence = [w for w in words if not w.lower() in stop_words]
filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(filtered_sentence)

#stemming: eliminar plurales
stemmer = SnowballStemmer("spanish")
singles_words = [stemmer.stem(plural) for plural in words]
print(singles_words)


