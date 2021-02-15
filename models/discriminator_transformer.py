import torch
from torch import nn
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import json
from typing import List, Dict, Optional
import os
import numpy as np
import logging

from models import Transformer


class DiscriminatorTransformer(Transformer):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None, out_dim=2):
        #super(DiscriminatorTransformer, self).__init__(model_name_or_path)
        super(Transformer, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name_or_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2, return_dict=True)
        self.config = self.model.config
        #self.config_class = self.model.config_class
        self.config_keys = ['max_seq_length']
        self.dtype = self.model.dtype
        self.config_class = self.model.config_class
        self.max_seq_length = max_seq_length

    def forward(self, features, label, attach_grad_fn=None):
        """Returns token_embeddings, cls_token"""
        embedding = self.tokenizer(features, return_tensors='pt', truncation=True,
                max_length=self.max_seq_length,padding=True,).to(self.model.device)
        #if attach_grad_fn is not None:
        #    t = torch.tensor(0.0, requires_grad=True)
        #    embedding.input_ids = embedding.input_ids.type(torch.float32)
        #    embedding.input_ids += t
        #    embedding.input_ids.backward()
        #    embedding.input_ids.grad.data.copy_(attach_grad_fn.grad.data)
        labels = self.get_target_tensor(torch.zeros(embedding["input_ids"].size()[0]), label)
        features = self.model(**embedding, labels=labels)
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 #Add space for special tokens
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, padding=True, return_tensors='pt')

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return DiscriminatorTransformer(model_name_or_path=input_path, **config)


    def encodeSentence(self,sentence):
        input_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding=True,
                #pad_to_max_length=True,
                return_tensors='pt'
            )

        return input_ids[0, :]


    def batch_encode_plus(self, sentences, verbose=True):
        train_input_ids = self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding=True,
                #pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )

        return train_input_ids


    def get_target_tensor(self, prediction, label):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """


        target_tensor = torch.tensor(label).to(self.model.device)
        return target_tensor.expand_as(prediction)
