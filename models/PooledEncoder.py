import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
import models
import json
from typing import List, Dict, Optional
import os
import numpy as np
import logging

class PooledEncoder(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None, out_dimension=None):
        super(PooledEncoder, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        #config.output_hidden_states = True
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.embedding_pooling = models.Pooling(self.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
        if out_dimension is not None:
            self.auto_model.base_model.pooling_layer = nn.Linear(self.get_word_embedding_dimension(), out_dimension)


        self.config = self.auto_model.config
        self.auto_model.config.output_attentions = True
        self.auto_model.config.output_hidden_states = True

    def forward(self, input_ids, attention_mask):
        """Returns token_embeddings, cls_token"""
        features = dict()
        features.update({'input_ids': input_ids, 'attention_mask': attention_mask})

        output_states = self.auto_model(**features)
        output_tokens = output_states[0]


        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token

        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        output_states += tuple(self.embedding_pooling(features)['sentence_embedding'])   #sentence embedding

        output_states_list = list(output_states)
        output_states_list[0] = self.auto_model.base_model.pooling_layer(output_states[0])
        output_states_list[1] = self.auto_model.base_model.pooling_layer(output_states[1])
        tup = tuple()
        for i, data in enumerate(output_states_list[2]):
            value = self.auto_model.base_model.pooling_layer(data)
            tup += tuple(value)
        output_states_list[2] = tup
        output_states = tuple(output_states_list)

        return output_states

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
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')

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
        return PooledEncoder(model_name_or_path=input_path, **config)

    def encode(self, sentences, verbose=True):
        train_input_ids = []
        if verbose:
            for text in tqdm(sentences):
                input_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )
                train_input_ids.append(input_ids)
        else:
            for text in sentences:
                input_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True
                )
                train_input_ids.append(input_ids)

        train_input_ids = torch.cat(train_input_ids, dim=0)
        return train_input_ids

    def encodeSentence(self,sentence):
        input_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_tensors='pt'
            )

        return input_ids[0, :]


