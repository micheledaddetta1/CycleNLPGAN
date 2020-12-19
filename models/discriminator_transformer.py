import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelWithLMHead
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
        super(DiscriminatorTransformer, self).__init__(model_name_or_path)
        self.layers = nn.Linear(self.get_word_embedding_dimension(), out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        features = super(DiscriminatorTransformer, self).forward(features)
        features = self.layers(features['sentence_embedding'])
        features = self.softmax(features)
        _, features = torch.max(features, 1)
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
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)


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
