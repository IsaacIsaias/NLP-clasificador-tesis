# NLP-clasificador-tesis

Se creó un dataset a partir de un proceso de scraping donde se extrajeron tesis de la Universidad Autónoma de México (UNAM) en el siguiente link: https://tesiunam.dgb.unam.mx/F?func=find-b-0&local_base=TES01. Se extrajeron primeramente 200 tesis de 5 carreras de esta universidad: Actuaría, Derecho, Economía, Psicología y Química Farmacéutico Biológica.

El dataset fue procesado con las siguientes tareas (dataset_tesis_procesado.csv):
- convertir a minúsculas
- tokenización
- eliminar palabras que no son alfanuméricas
- elimanr palabras vacías
- stemming: eliminar plurales

# Cita

"Esta base de datos/modelo se ha creado/entrenado en el marco del Hackathon 2022 de PLN en Español organizado por Somos NLP patrocinado por Platzi, Paperspace y Hugging Face: https://huggingface.co/hackathon-pln-es."

Miembros de equipo (user de Hugging Face):
- Isacc Isahias López López (MajorIsaiah)
- Dionis López (inoid)
- Yisel Clavel Quintero (clavel)
- Ximyer Yeraldin López López (Ximyer)
