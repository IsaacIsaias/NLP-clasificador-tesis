import sys,os,argparse,time
from huggingface_hub import notebook_login
import transformers
from datasets import load_dataset, load_metric, Features, Value, ClassLabel,  Dataset
import datasets
import random
import pandas as pd
from sklearn.utils import shuffle
from IPython.display import display, HTML

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


parser = argparse.ArgumentParser()
parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="Name of spanish bert model: BETO.")
parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--num_train_epochs",
                    default=6,
                    type=int,
                    help="Total number of training epochs to perform.")


args = parser


DIRECTORY_ADDRES = 'datasets'
FILE_NAME = 'preprocessed_data_pipe.csv'
#
#
df = pd.read_csv( DIRECTORY_ADDRES + os.path.sep + FILE_NAME, names =  ['texto','autor_nombre','autor_apellido','titulo','año','carrera'], delimiter="|", header=None,skiprows = 1)

columns = ['titulo', 'carrera']
df = pd.DataFrame(df, columns=columns)
# for index, row in df.iterrows():
#    if index > 0:
#      print(row['autor'], row['titulo'],row['año'],row['carrera'])

###Transform in dataset data strctured
#Shuffle elements

#Bibliografy:
#
# https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
#
# Shuffle the DataFrame rows
df = df.sample(frac = 1)
string_text = str(df['carrera'].values)
print (set(df['carrera'].to_list()))
print (df['titulo'])

classSetOfTesisClasiication = set(df['carrera'].to_list())
sizeOfClassClassification = len (classSetOfTesisClasiication)
class_names = list(classSetOfTesisClasiication)


#dataset_from_pandas.cast_column("carrera", ClassLabel(num_classes=sizeOfClassClassification, names=class_names, names_file=None, id=None))

#Size of rows

#Bibliografy:
#  https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html
#  https://huggingface.co/docs/datasets/loading
#  https://towardsdatascience.com/my-experience-with-uploading-a-dataset-on-huggingfaces-dataset-hub-803051942c2d
#
#
class_names=["Enseñanza de Inglés","Español","Historia"]
emotion_features = Features({'texto': Value('string'), 'carrera': ClassLabel(names=class_names)})

###Split in test, dev and train
dataset = Dataset.from_pandas(df)

# def process(ex):
#     ex['carrera']: emotion_features['carrera'].names.index(ex['carrera'])
#     return ex
# dataset = dataset.map(process)

dataset = dataset.train_test_split(test_size=0.1)
print (dataset["train"][0])
print (dataset["test"][0])

spanish_models = {'BETO':"hiiamsid/BETO_es_binary_classification",'Bertin':"bertin-project/bertin-base-xnli-es"}
#################################################################################################
# Example:
#
# https://benjad.github.io/2020/08/04/clasificador-sentimiento-BERT/
#
# Resources
# Spanish Modelss
#
#
# bertin-project/bertin-roberta-base-spanish https://huggingface.co/bertin-project/bertin-roberta-base-spanish
#
# bertin-project/bertin-base-xnli-es https://huggingface.co/bertin-project/bertin-base-xnli-es
#
# PlanTL-GOB-ES/roberta-large-bne https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne
#
# PlanTL-GOB-ES/roberta-base-bne https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
#
# mrm8488/electricidad-base-discriminator https://huggingface.co/mrm8488/electricidad-base-discriminator
#
#flax-community/spanish-t5-small https://huggingface.co/flax-community/spanish-t5-small
#################################################################################################

spanish_models = {'BETO':"hiiamsid/BETO_es_binary_classification",
                  'ROBERTA_E':"bertin-project/bertin-roberta-base-spanish",
                  'BERTIN': "bertin-project/bertin-base-xnli-es",
                  'ROBERT_GOB':"PlanTL-GOB-ES/roberta-large-bne",
                  'ROBERT_GOB_PLUS':"PlanTL-GOB-ES/roberta-base-bne",
                  'ELECTRA':"mrm8488/electricidad-base-discriminator",
                  'ELECTRA_SMALL':"flax-community/spanish-t5-small"
                  }

#"hiiamsid/BETO_es_binary_classification",cache_dir =

tokenizer = AutoTokenizer.from_pretrained(spanish_models['BETO'] ,use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(spanish_models['BETO'])


# tokenizer = AutoTokenizer.from_pretrained( "." + os.path.sep + "models" + os.path.sep + "beto" + os.path.sep,use_fast=False)
# model = AutoModelForSequenceClassification.from_pretrained("." + os.path.sep + "models" + os.path.sep + "beto" + os.path.sep)

text = "Replace me by any text you'd like."

def preprocess_function(examples):
    return tokenizer(examples["titulo"], truncation=True)

tokenized_snpanish = dataset.map(preprocess_function, batched=True)

# Use DataCollatorWithPadding to create a batch of examples.
# It will also dynamically pad your text to the length of the
# longest element in its batch, so they are a uniform length.
# While it is possible to pad your text in the tokenizer function by setting padding=True,
# dynamic padding is more efficient.

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Fine-tune with Trainer

#model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(emotion_features['carrera'].names))

#Define Trainer argument

from datasets import load_metric
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./finetunigmodel",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    gradient_accumulation_steps=2
)

training_args = TrainingArguments(output_dir="./finetunigmodel", evaluation_strategy="epoch")

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"],

    compute_metrics=compute_metrics,

)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

trainer.train()

notebook_login()

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

dataset

dataset["train"][0]
dataset["test"][0]




def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(dataset["train"])

metric

import numpy as np

#ou can call its compute method with your predictions
# and labels directly and it will return a dictionary with the metric(s) value:

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)

#PREPROCESSING DATA

# Before we can feed those texts to our model, we need to preprocess them. This is done by
# a 🤗 Transformers Tokenizer which will (as the name indicates) tokenize the inputs
# (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put
# it in a format the model expects, as well as generate the other inputs that model requires.
#
# To do all of this, we instantiate our tokenizer with the AutoTokenizer.from_pretrained method, which will ensure:
#
#     we get a tokenizer that corresponds to the model architecture we want to use,
#     we download the vocabulary used when pretraining this specific checkpoint.
#
# That vocabulary will be cached, so it's not downloaded again the next time we run the cell.

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

tokenizer("Hello, this one sentence!", "And this sentence goes with it.")

# Depending on the model you selected, you will see different keys in the dictionary returned by
# the cell above. They don't matter much for what we're doing here (just know they are required by
# the model we will instantiate later), you can learn more about them in this tutorial
# if you're interested.
#
# To preprocess our dataset, we will thus need the names of the columns containing the sentence(s).
# The following dictionary keeps track of the correspondence task to column names:


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

# We can them write the function that will preprocess our samples.
# We just feed them to the tokenizer with the argument truncation=True.
# This will ensure that an input longer that what the model selected can handle
# will be truncated to the maximum length accepted by the model.

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


# This function works with one or several examples.
# In the case of several examples, the tokenizer will return a list of lists for each key:

preprocess_function(dataset['train'][:5])

# To apply this function on all the sentences (or pairs of sentences) in our dataset,
# we just use the map method of our dataset object we created earlier.
# This will apply the function on all the elements of all the splits in dataset, so our training,
# validation and testing data will be preprocessed in one single command.

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Even better, the results are automatically cached by the 🤗 Datasets library to avoid
# spending time on this step the next time you run your notebook.
# The 🤗 Datasets library is normally smart enough to detect when the function
# you pass to map has changed (and thus requires to not use the cache data).
# For instance, it will properly detect if you change the task in the
# first cell and rerun the notebook. 🤗 Datasets warns you when it
# uses cached files, you can pass load_from_cache_file=False in the call
# to map to not use the cached files and force the preprocessing to be applied again.

# Note that we passed batched=True to encode the texts by batches together.
# This is to leverage the full benefit of the fast tokenizer we loaded earlier,
# which will use multi-threading to treat the texts in a batch concurrently.


#Fine-tuning the model

# Now that our data is ready, we can download the pretrained model and fine-tune it.
# Since all our tasks are about sentence classification, we use the AutoModelForSequenceClassification
# class. Like with the tokenizer, the from_pretrained method will download and cache the model for us.
# The only thing we have to specify is the number of labels for our problem
# (which is always 2, except for STS-B which is a regression problem and MNLI where we have 3 labels):

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# The warning is telling us we are throwing away some weights (the vocab_transform and vocab_layer_norm
# layers) and randomly initializing some other (the pre_classifier and classifier layers).
# This is absolutely normal in this case, because we are removing the head used to pretrain the
# model on a masked language modeling objective and replacing it with a new head for which
# we don't have pretrained weights, so the library warns us we should fine-tune this model ' \
# 'before using it for inference, which is exactly what we are going to do.
#
# To instantiate a Trainer, we will need to define two more things. The most important is the
# TrainingArguments, which is a class that contains all the attributes to customize the training.
# It requires one folder name, which will be used to save the checkpoints of the model, and all
# other arguments are optional:

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=True,
)


# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use
# the batch_size defined at the top of the notebook and customize the number of epochs for
# training, as well as the weight decay. Since the best model might not be the one at the end of
# training, we ask the Trainer to load the best model it saved (according to metric_name) at the
# end of training.
#
# The last argument to setup everything so we can push the model to the Hub regularly during training.
# Remove it if you didn't follow the installation steps at the top of the notebook. If you want to
# save your model locally in a name that is different than the name of the repository it will be pushed,
# or if you want to push your model under an organization and not your name space, use the hub_model_id argument
# to set the repo name (it needs to be the full name, including your namespace: for instance "sgugger/bert-finetuned-mrpc" or "huggingface/bert-finetuned-mrpc").
#
# The last thing to define for our Trainer is how to compute the metrics from the predictions.
# We need to define a function for this, which will just use the metric we loaded earlier, the only
# preprocessing we have to do is to take the argmax of our predicted logits (our just squeeze the last
# axis in the case of STS-B):

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

#Then we just need to pass all of this along with our datasets to the Trainer:

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# You might wonder why we pass along the tokenizer when we already preprocessed our data.
# This is because we will use it once last time to make all the samples we gather the same
# length by applying padding, which requires knowing the model's '
# preferences regarding padding (to the left or right? with which token?).
# The tokenizer has a pad method that will do all of this right for us, and the Trainer will use it.
# You can customize this part by defining and passing your own data_collator which will receive the
# samples like the dictionaries seen above and will need to return a dictionary of tensors.


trainer.train()

# We can check with the evaluate method that our
# Trainer did reload the best model properly (if it was not the last one):

trainer.evaluate()

# To see how your model fared you can compare it to the GLUE Benchmark leaderboard.
#
# You can now upload the result of the training to the Hub, just execute this instruction:

trainer.push_to_hub()

# You can now share this model with all your friends, family, favorite pets: they can all load it with
# the identifier "your-username/the-name-you-picked" so for instance:

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("sgugger/my-awesome-model")



#We can now finetune our model by just calling the train method:



#Bibliografy:
# Pipelines for inference
#https://huggingface.co/docs/transformers/pipeline_tutorial

# Fine-tune a pretrained model
#https://huggingface.co/docs/transformers/training

# Preprocess
# Uff durisimo

#https://huggingface.co/docs/transformers/preprocessing

#Text classification
#https://huggingface.co/docs/transformers/tasks/sequence_classification

# How to fine-tune a model for common downstream tasks
#https://huggingface.co/docs/transformers/custom_datasets

#Create a Dataset
#https://huggingface.co/docs/datasets/about_dataset_load
#https://huggingface.co/docs/datasets/dataset_script


#Paperspace HPC
#https://www.youtube.com/watch?v=bZ2bY5w7s10

