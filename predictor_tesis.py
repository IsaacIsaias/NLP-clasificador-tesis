import torch
from transformers import  BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AdamW
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
import pickle
from transformers import TextClassificationPipeline


def preprocessingtext(_text, tokenizer):
    if _text == "" or _text == None or tokenizer == None:
        raise Exception("Not  text or error")
    else:
        encoded_doc = tokenizer.encode_plus(_text,
                                            add_special_tokens=True, max_length=115,
                                            truncation=True, pad_to_max_length=True)
        return (torch.tensor(encoded_doc['input_ids']),
                torch.tensor(encoded_doc['attention_mask']))

def classifyText(_text, model_name):
     #if exist model
     spanish_models = {'BETO': "hiiamsid/BETO_es_binary_classification",
                       'ROBERTA_E': "bertin-project/bertin-roberta-base-spanish",
                       'BERTIN': "bertin-project/bertin-base-xnli-es",
                       'ROBERT_GOB': "PlanTL-GOB-ES/roberta-large-bne",
                       'ROBERT_GOB_PLUS': "PlanTL-GOB-ES/roberta-base-bne",
                       'ELECTRA': "mrm8488/electricidad-base-discriminator",
                       'ELECTRA_SMALL': "flax-community/spanish-t5-small"
                       }

     if model_name != "" and model_name != None and model_name in spanish_models:
              path = os.path.join("./finetunigmodel", spanish_models[model_name])
         #if os.path.exists(path):
              ###Transform in token data
              tokenizer = AutoTokenizer.from_pretrained(spanish_models[model_name], use_fast=False)
              X_val_inputs, X_val_masks = preprocessingtext(_text,tokenizer)
              t0 = time.time()

              # Deserialization of the file
              #file = open(path + os.path.sep + 'classIndexAssociation.pkl', 'rb')
              #new_model = pickle.load(file)

              #sizeOfClass = len(new_model)

              model = AutoModelForSequenceClassification.from_pretrained(
                   'hackathon-pln-es/unam_tesis_BETO_finnetuning', num_labels=5, output_attentions=False,
                  output_hidden_states=False)
              #Bibliografy from:
              #
              #  https://huggingface.co/docs/transformers/main_classes/output
              #
              inputs = tokenizer(_text, return_tensors="pt")
              labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
              outputs = model(**inputs, labels=labels)

              loss, logits = outputs[:2]

              #Transform in array
              logits = logits.detach().cpu().numpy()

              #Get max element and position
              result = logits.argmax()
              return new_model[result]

              #Example from
              #
              #
              #
              # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
              # # Put the model in evaluation mode
              # classificationResult = pipe(_text)
              # if  classificationResult[0]  != None and len (classificationResult[0]) > 0:
              #     #Order the result with more close to 1
              #     classificationResult[0].sort(reverse=True, key=lambda x:x['score'])
              #     # Return the text clasification
              #     keyClass = classificationResult[0][0]['label']
              #     keyClass = keyClass.replace("LABEL_","").strip()
              #     if  keyClass.isnumeric():
              #       return new_model[ int (keyClass)]
              #     else:
              #         raise Exception("Not exist class info")
                  # model.eval()
                  # outputs = model(X_val_inputs,
                  #                 token_type_ids=None,
                  #                 attention_mask=X_val_masks)
                  #
                  # # The "logits" are the output values
                  # # prior to applying an activation function
                  # logits = outputs[0]
                  #
                  # # Move logits and labels to CPU
                  # logits = logits.detach().cpu().numpy()
                  #
                  # sorted_tuples = sorted(logits.items(), key=lambda item: item[1])
                  # #Return the text clasification
                  # keyClass = sorted_tuples.keys()[0]
                  # return new_model[keyClass]

         # else:
         #     raise Exception("Not exist model info")
     else:
        raise Exception("Not exist model info")
     return "Text"



print(classifyText('Boosting con Ã¡rboles de decisiÃ³n y random forest','BETO'))