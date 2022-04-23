import torch
from transformers import  BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, AutoConfig, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, Features, Value, ClassLabel,  Dataset
from sklearn.model_selection import train_test_split
import torch.optim
import numpy as np
import pandas as pd
import sys,os,argparse,time
import time
import datetime
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
from transformers import TextClassificationPipeline

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--bert_model", default='BETO', type=str, help="Name of spanish bert model: BETO, ROBERTA_E, BERTIN, ROBERT_GOB \
ROBERT_GOB_PLUS, ELECTRA, ELECTRA_SMALL")

parser.add_argument("--train_field", default='titulo', type=str, help=" titulo, texto,  both, both-rev i.e., Name of UNAM thesis field to tuning the huggingface Spanish model")
parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--num_train_epochs",
                    default=3,
                    type=int,
                    help="Total number of training epochs to perform.")

parser.add_argument("--scope",
                    default="cpu",
                    type=str,
                    help="Scope of training gpu or cpu")

parser.add_argument("--exprm",
                    default="",
                    type=str,
                    help="Name for module finetuning creation e.g, BETO-both: Pretrained BETO model with Unam tesis texto and title concat")
parser.add_argument('--typedata', action='store_true', help='Indicate wich dataset use i.e., dataset_tesis.csv (by default) or dataset_tesis_procesado.csv')
args  = parser.parse_args()

# Select cpu or cuda
run_on = args.scope
if run_on == None or run_on == "" or (run_on != "gpu" and run_on != "cpu") :
    run_on = 'cpu'
device = torch.device(run_on)

#Bibliografy
#
# https://benjad.github.io/2020/08/04/clasificador-sentimiento-BERT/
#
#


# Load the dataset into a pandas dataframe.
#df = pd.read_csv('/reviewsclean.csv', header=0)
DIRECTORY_ADDRES = 'datasets'
#FILE_NAME = 'dataset_tesis.csv'
FILE_NAME = 'expreprocessed_data_pipe.csv'
#FILE_NAME = 'dataset_tesis.csv'
columns_name = ['texto', 'autor_nombre', 'autor_apellido', 'titulo', 'año', 'carrera']

if args.typedata != False :
    columns_name = ['texto','titulo','carrera']
    FILE_NAME = 'dataset_tesis_procesado.csv'
#encoding='ISO-8859-1'
df = pd.read_csv( DIRECTORY_ADDRES + os.path.sep + FILE_NAME, names =columns_name , encoding='UTF-8', sep='|', engine='python', header=None,skiprows = 1)

train_option = ['titulo','texto','both','both-rev']
train_field = args.train_field
if train_field == "" or train_field == None or ( not train_field in train_option):
    print("WARRRING BAT MODEL NAME INICIALIZATION SET BETO as DEFAULT")
    train_option =train_option[0]

if train_field == "both":
    reviews = df["texto"].astype(str) + " " + df["titulo"]
elif train_field == "both-rev":
    reviews = df["titulo"].astype(str) + " " + df["texto"]
else:
  reviews = df[train_field]

df["process_texto"] = reviews

####Split text because their size is more large than 200
#
# Bibliografy:
#
#   https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d
#
#   Articles :
#      Hierarchical Transformers for Long Document Classification
#         - https://arxiv.org/abs/1910.10781
#
#     DocBERT: BERT for Document Classification
#         - https://arxiv.org/abs/1904.08398
#  WARNING!!!: Transform in a list of token list with 200 as max size
#
def get_split(text1):
  l_total = []
  l_parcial = []
  if len(text1.split())//150 >0:
    n = len(text1.split())//150
  else:
    n = 1
  for w in range(n):
    if w == 0:
      l_parcial = text1.split()[:200]
      l_total.append(" ".join(l_parcial))
    else:
      l_parcial = text1.split()[w*150:w*150 + 200]
      l_total.append(" ".join(l_parcial))
  return l_total

df['text_split'] = df["process_texto"].apply(get_split)

print (df.head())

#Evaluate data in file

#reviews = df['text_split']


#Class filed in datasets
sentiment = df['carrera']

class_names = list(set(sentiment))
sizeOfClass = len(set(class_names))
emotion_features = Features({'texto': Value('string'), 'carrera': ClassLabel(names=class_names)})
classIndex = [ emotion_features['carrera'].names.index(x) for x in emotion_features['carrera'].names]

idIndexClassTuple = dict(zip(classIndex,class_names))
classNameIndexTuple =  dict(zip(class_names,classIndex))

sentiment = sentiment.replace(class_names,classIndex)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(reviews,
sentiment, stratify=sentiment, test_size=0.2, random_state=42)

# Report datasets lenghts
print('Training set length : {}'.format(len(X_train)))
print('Validation set length : {}'.format(len(X_val)))

spanish_models = {'BETO':"hiiamsid/BETO_es_binary_classification",
                  'ROBERTA_E':"bertin-project/bertin-roberta-base-spanish",
                  'BERTIN': "bertin-project/bertin-base-xnli-es",
                  'ROBERT_GOB':"PlanTL-GOB-ES/roberta-large-bne",
                  'ROBERT_GOB_PLUS':"PlanTL-GOB-ES/roberta-base-bne",
                  'ELECTRA':"mrm8488/electricidad-base-discriminator",
                  'ELECTRA_SMALL':"flax-community/spanish-t5-small"
                  }


#Tokenization

# tokenizer = BertTokenizer.from_pretrained("pytorch/",
#             do_lower_case=True)

model_name = args.bert_model

if model_name == None or model_name == "" or not model_name in spanish_models :
    print("WARRRING BAT MODEL NAME INICIALIZATION SET BETO as DEFAULT")
    model_name = 'BETO'

tokenizer = AutoTokenizer.from_pretrained(spanish_models[model_name] ,use_fast=False)

def preprocessing(dataset):
    input_ids = []
    attention_mask = []
    for doc in dataset:
        encoded_doc = tokenizer.encode_plus(doc,
                   add_special_tokens=True, max_length=115,
                   truncation=True,pad_to_max_length=True)
        input_ids.append(encoded_doc['input_ids'])
        attention_mask.append(encoded_doc['attention_mask'])
    return (torch.tensor(input_ids),
           torch.tensor(attention_mask))

def preprocessingtext(_text, tokenizer):
     if _text == "" or _text == None or tokenizer == None:
         raise Exception("Not  text or error")
     else:
         encoded_doc = tokenizer.encode_plus(_text,
                                             add_special_tokens=True, max_length=115,
                                             truncation=True, pad_to_max_length=True)
         return (torch.tensor(encoded_doc['input_ids']),
             torch.tensor(encoded_doc['attention_mask']))



# Apply preprocessing to dataset
X_train_inputs, X_train_masks = preprocessing(X_train)
X_val_inputs, X_val_masks = preprocessing(X_val)

# Report max n° tokens in a sentence
max_len = max([torch.sum(sen) for sen in X_train_masks])
print('Max n°tokens in a sentence: {0}'.format(max_len))

#Luego creamos los dataloaders de PyTorch para el dataset de entrenamiento y de validación.

# Data loaders

batch_size = args.train_batch_size
if batch_size <= 0 or batch_size == None:
    print("WARRRING BAT BATCH SIZE INICIALIZATION SET BETO as DEFAULT")
    batch_size = 32

y_train_labels = torch.tensor(y_train.values)
y_val_labels = torch.tensor(y_val.values)

#Set experiment name
experimentName = ""
if args.exprm != None and args.exprm != "":
    experimentName = args.exprm


def dataloader(x_inputs, x_masks, y_labels):
    data = TensorDataset(x_inputs, x_masks, y_labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler,
                 batch_size=batch_size,
                 num_workers=0)
    return dataloader

train_dataloader = dataloader(X_train_inputs, X_train_masks,
                   y_train_labels)
val_dataloader = dataloader(X_val_inputs, X_val_masks,
                 y_val_labels)


# Ahora establecemos los valores aleatorios, de manera de que nuestros resultados sean reproducibles.
# También cargamos el modelo, el optimizador, definimos los epochs y el scheduler en PyTorch.

# set random seed
def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
set_seed(42)

# Create model and optimizer
#model = AutoModelForSequenceClassification.from_pretrained(spanish_models['BETO'])


#Fix model class rearrange problem
#Bibliografy:
#
#  https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/21
#

config = AutoConfig.from_pretrained(spanish_models[model_name])
config.num_labels = sizeOfClass
config.output_hidden_states=False
model = AutoModelForSequenceClassification.from_config(config)

# model = AutoModelForSequenceClassification.from_pretrained(
#         , num_labels=sizeOfClass, output_attentions=False,
#          output_hidden_states=False)

#Replace for avoiding error in last final class number
#Bibliografy:
#
#  https://discuss.huggingface.co/t/replacing-last-layer-of-a-fine-tuned-model-to-use-different-set-of-labels/12995/5
#



model.config.id2label = idIndexClassTuple
model.config.label2id = classNameIndexTuple
model.config._num_labels = sizeOfClass ## replacing 9 by 13
model.config.num_labels = sizeOfClass


optimizer = AdamW(model.parameters(),
                  lr = 4e-5,
                  eps = 1e-6
                  )

if run_on == 'cuda':
    model.cuda()

# Define number of epochs

epochs = args.num_train_epochs
if epochs <= 0:
    print("WARRRING BAT BATCH SIZE INICIALIZATION SET BETO as DEFAULT")
    batch_size = 32

total_steps = len(train_dataloader) * epochs


##### STUDY THAT #######
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps = 0,
            num_training_steps = total_steps)


##Definimos una función para formatear el tiempo y otra para calcular la exactitud.

#fuction to format time
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

#function to compute accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#Por último definimos la función que se encargará de entrenar
# el modelo y también de entregar los resultados en el set de validación.

eval_losses=[]
eval_accu=[]
eval_f1=[]
# function to train the model

def training(n_epochs, training_dataloader,
             validation_dataloader, labels_tag):
    # ========================================
    #               Training
    # ========================================
    print('======= Training =======')
    for epoch_i in range(0, n_epochs):
        # Perform one full pass over the training set
        print("")
        print('======= Epoch {:} / {:} ======='.format(
            epoch_i + 1, epochs))
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        # Put the model into training mode.
        model.train()
        # For each batch of training data
        for step, batch in enumerate(training_dataloader):
            batch_loss = 0
            # Unpack this training batch from dataloader
            #   [0]: input ids, [1]: attention masks,
            #   [2]: labels
            b_input_ids, b_input_mask, b_labels = tuple(
                t.to(device) for t in batch)

            # Clear any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # pull loss value out of the output tuple
            loss = outputs[0]
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           1.0)

            # Update parameters
            # ¿take a step using the computed gradient
            optimizer.step()
            scheduler.step()

            print('batch loss: {0} | avg loss: {1}'.format(
                batch_loss, total_loss / (step + 1)))
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)



        print("")
        print("  Average training loss: {0:.2f}".
              format(avg_train_loss))
        print("  Training epoch took: {:}".format(
            format_time(time.time() - t0)))

        eval_losses.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch,
        # measure accuracy on the validation set.

        print("")
        print("======= Validation =======")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        all_logits = []
        all_labels = []
        # Evaluate data for one epoch
        for step, batch in enumerate(validation_dataloader):
            # Add batch to device
            # Unpack this training batch from our dataloader.
            #   [0]: input ids, [1]: attention masks,
            #   [2]: labels
            b_input_ids, b_input_mask, b_labels = tuple(
                t.to(device) for t in batch)

            # Model will not to compute gradients
            with torch.no_grad():
                # Forward pass
                # This will return the logits
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # The "logits" are the output values
            # prior to applying an activation function
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            b_labels = b_labels.to('cpu').numpy()

            # Save batch logits and labels
            # We will use thoses in the confusion matrix
            predict_labels = np.argmax(
                logits, axis=1).flatten()
            all_logits.extend(predict_labels.tolist())
            all_labels.extend(b_labels.tolist())

            # Calculate the accuracy for this batch
            tmp_eval_accuracy = flat_accuracy(
                logits, b_labels)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

        # Report the final accuracy for this validation run.
        f1 = metrics.f1_score(all_labels, all_logits, labels=list(labels_tag), average='macro')

        print("  Accuracy: {0:.2f}".
              format(eval_accuracy / (step + 1)))
        print("  F1-Macro: {0:.2f}".
              format(f1 / (step + 1)))
        print("  Validation took: {:}".format(
            format_time(time.time() - t0)))

        eval_accu.append(eval_accuracy / (step + 1))
        eval_f1.append(f1 / (step + 1))


    # print the confusion matrix"
    conf = confusion_matrix(
        all_labels, all_logits, normalize='true')
    print(conf)
    print("")
    print("Training complete")


# call function to train the model
training(epochs, train_dataloader, val_dataloader, labels_tag=idIndexClassTuple.keys())

#Save the Model
path = os.path.join("./finetunigmodel", spanish_models[model_name] + experimentName)
if not os.path.exists(path):
     # Handle the errors
    try:
        # Create the directory in the path
        os.makedirs(path, exist_ok=True)
        print("Directory %s Created Successfully" % spanish_models[model_name])
    except OSError as error:
        print("Directory %s Creation Failed" % spanish_models[model_name])

modelToSaveIn = path

model.save_pretrained(modelToSaveIn)
###Save in File the Correspondence between
## Number and Class to load in clasification processs
## with system


print('Weights before pickling')

# Open a file to write bytes
p_file = open(modelToSaveIn + os.path.sep + 'classIndexAssociation.pkl', 'wb')

# Pickle the object
pickle.dump(idIndexClassTuple, p_file)
p_file.close()


# Deserialization of the file
file = open(modelToSaveIn + os.path.sep + 'classIndexAssociation.pkl', 'rb')
new_model = pickle.load(file)

print('Weights after pickling', new_model)


#plot accuracy,f1 and loss training processing data
#
#  Bibliografy:
#
#  https://androidkt.com/calculate-total-loss-and-accuracy-at-every-epoch-and-plot-using-matplotlib-in-pytorch/
#
plt.plot(eval_accu,'-o')
plt.plot(eval_f1,'-o')
plt.plot(eval_losses,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Accur','F1','Loss'])
plt.title('Accur, F1 and Loss in epoch trainig')
plt.savefig(   str(time.time()) + experimentName +'.png', bbox_inches='tight')
plt.show()


