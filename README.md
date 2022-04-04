# NLP-clasificador-tesis

Se creó un dataset (dataset_tesis.csv) a partir de un proceso de scraping donde se extrajeron tesis de la Universidad Autónoma de México (UNAM) en el siguiente link: https://tesiunam.dgb.unam.mx/F?func=find-b-0&local_base=TES01. Se extrajeron primeramente 200 tesis de 5 carreras de esta universidad: Actuaría, Derecho, Economía, Psicología y Química Farmacéutico Biológica. De estas se extrajo: introduscción, nombre del autor, apellidos de autor, título de la tesis y la carrera. 

El dataset fue procesado con las siguientes tareas (dataset_tesis_procesado.csv):
- convertir a minúsculas
- tokenización
- eliminar palabras que no son alfanuméricas
- elimanr palabras vacías
- stemming: eliminar plurales

Ambos datasets, el creado con el texto original y el procesado fueron subidos a https://huggingface.co/hackathon-pln-es/.

Se hizo finetunning a partir de modelos de transformes del estado del arte existentes en Hugging Face, los siguientes fueron subidos a https://huggingface.co/hackathon-pln-es/:
- 'BETO': "inoid/unam_tesis_BETO_finnetuning",
- 'ROBERTA_E': "inoid/unam_tesis_ROBERTA_es_finnetuning"
- 'BERTIN': "inoid/unam_tesis_BERTIN_finnetuning"
- 'ROBERT_GOB': "inoid/unam_tesis_ROBERTA_GOB_finnetuning",
- 'ROBERT_GOB_PLUS': "inoid/unam_tesis_ROBERT_GOB_PLUS_finnetuning",
- 'ELECTRA': "inoid/unam_tesis_ELECTRA_finnetuning",
- 'ELECTRA_SMALL': "inoid/unam_tesis_ELECTRA_SMALL_finnetuning"

# Cita

"Esta base de datos/modelo se ha creado/entrenado en el marco del Hackathon 2022 de PLN en Español organizado por Somos NLP patrocinado por Platzi, Paperspace y Hugging Face: https://huggingface.co/hackathon-pln-es."

Miembros de equipo (user de Hugging Face):
- Isacc Isahias López López (MajorIsaiah)
- Dionis López (inoid)
- Yisel Clavel Quintero (clavel)
- Ximyer Yeraldin López López (Ximyer)
