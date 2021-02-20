#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
import transformers
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.batch_encode_plus(["Hello, my dog is cute", "This is a long sentence because I want understand padding"])).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
'''
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, MarianMTModel, MarianTokenizer
from transformers import MarianTokenizer, MarianMTModel
from models import Pooling, EncDecModel


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from transformers import pipeline, set_seed
generator = pipeline('translation_en_to_fr', model='t5-base')
set_seed(42)
res = generator("Hello, I'm a language model" , max_length=30)
model_1 = EncDecModel('Helsinki-NLP/opus-mt-en-it')
model_2 = EncDecModel('Helsinki-NLP/opus-mt-en-de')
outputs_1 = model_1(["Studies have been shown that owning a dog is good for you"], ["Studi hanno dimostrato che avere un cane fa bene"],True)
outputs_2 = model_2(["Studies have been shown that owning a dog is good for you"], ["Studien haben gezeigt, dass der Besitz eines Hundes gut für Sie ist"],True)

print(outputs_2[1]==outputs_1[1])
outputs2 = model_1(input_ids=input_ids, return_dict=True, training=True)

a= outputs.encoder_last_hidden_state
b= outputs2.encoder_last_hidden_state
last_hidden_states = outputs.last_hidden_state

model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-it')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-it')
model.to(device)
model.train()

train_loader = DataLoader(["Hi, I'm an university student"], batch_size=1, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        batch = tokenizer.batch_encode_plus(batch,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                #pad_to_max_length=True,
                truncation=True)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = torch.tensor([1]).to(device)
        outputs = model.base_model.encoder(input_ids, attention_mask=attention_mask)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
'''