import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, MarianTokenizer, MarianMTModel
import json
from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import logging
from models.Pooling import Pooling
from transformers.tokenization_utils import BatchEncoding



class EncDecModel(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, task="translation", model_args: Dict = {}, cache_dir: Optional[str] = None, freeze_encoder=False):
        super(EncDecModel, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.model = MarianMTModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

        self.tokenizer = MarianTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        self.config = self.model.config
        self.config_class = self.model.config_class
        #self.device = self.model.device
        self.dtype = self.model.dtype
        self.task = task

        #self.output_attentions = True
        #self.output_hidden_states = True
        #self.config.output_attentions = True
        #self.config.output_hidden_states = True

        self.freeze_encoder = freeze_encoder

        self.add_pooling_layer()

    def forward(self, sentences, target_sentences, partial_value=False):

        embeddings = self.batch_encode_plus(sentences, padding=True, verbose=False)
        embeddings = embeddings.to(self.model.device)
        if self.task == "translation":
            decoder_input_ids = self.batch_encode_plus(target_sentences,  padding=True, verbose=False).input_ids.to(self.model.device)  # Batch size 1
            outputs = self.model(input_ids=embeddings.input_ids, labels=decoder_input_ids, return_dict=True)

            output_sentences = self.model.generate(**embeddings)
            output_sentences = self.decode(output_sentences)
        else:
            output_sentences = self.generate(sentences)
            output_sentences = self.decode(output_sentences)


        if partial_value:
            '''
            embeddings.update({'output_hidden_states': True})
            partial = self.model.base_model.encoder(**embeddings)
            partial = partial[0][:, 0, :]  # CLS token is first token

            if self.task == "reconstruction":
                cls_tokens = partial[0][:, 0, :]  # CLS token is first token
                embeddings.update({'cls_token_embeddings': cls_tokens})
                partial = self.embedding_pooling(embeddings)
            del embeddings
            '''
            sentence_embedding = torch.zeros([len(sentences), self.get_word_embedding_dimension()], dtype=torch.float32).to(self.model.device)
            for i in range(len(sentences)):
                a = outputs.encoder_last_hidden_state[i]
                sentence_embedding[i] = self.embedding_pooling(outputs.encoder_last_hidden_state[i])["sentence_embedding"]
            return output_sentences, sentence_embedding, outputs.loss
        else:
            del embeddings

            return output_sentences, outputs.loss

    def get_word_embedding_dimension(self) -> int:
        return self.config.hidden_size

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

        #with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
        #    json.dump(self.get_config_dict(), fOut, indent=2)


    @staticmethod
    def load(input_path: str):
        #with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
        #    config = json.load(fIn)

        return EncDecModel(model_name_or_path=input_path) #, **config)


    def encodeSentence(self, sentence):
        logging.info("Trainer - encoding sentence")
        train_input_ids = []
        input_ids = self.tokenizer.encode(
                sentence,
                return_tensors='pt'
            )
        return input_ids[0, :]


    def generate(self, text):
        encod = self.tokenizer.prepare_translation_batch(text).to(self.model.device)
        output = self.model.generate(**encod)
        return output


    def get_encoder(self):
        return self.model.get_encoder()


    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        #return self.dest_tokenizer.batch_decode(tokens, skip_special_tokens=True)


    def train(self, mode=True):
        self.training = mode
        self.model.training = mode
        self.model.base_model.training = mode
        self.model.base_model.encoder.training = mode
        self.model.base_model.decoder.training = mode
        self.model.train(mode)
        self.model.base_model.train(mode)
        if self.freeze_encoder is True:
            self.model.base_model.encoder.training = False
            self.model.base_model.encoder.eval()
            self.model.base_model.decoder.train(mode)
            for param in self.model.base_model.encoder.parameters():
                param.requires_grad = False

    def eval(self):
        self.training = False
        self.model.training = False
        self.model.base_model.training = False
        self.model.base_model.encoder.training = False
        self.model.base_model.decoder.training = False
        self.model.eval()


    def redefine_config(self):
        self.config.architectures[0] = "MixedModel"
        self.config.encoder_attention_heads = self.model.base_model.encoder.config.num_attention_heads
        #self.config.hidden_size = None
        #self.config.hidden_size = self.model.base_model.encoder.config.hidden_size
        self.config.encoder_layers = self.model.base_model.encoder.config.num_hidden_layers


    def encode(self, sentences, verbose=True):
        train_input_ids = []
        if verbose:
            for text in tqdm(sentences):
                input_ids = self.tokenizer.encode(
                    text,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding=True,
                    #pad_to_max_length=True,
                    truncation=True
                )
                train_input_ids.append(input_ids)
        else:
            for text in sentences:
                input_ids = self.tokenizer.encode(
                    text,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding=True,
                    #pad_to_max_length=True,
                    truncation=True
                )

                train_input_ids.append(input_ids)

        train_input_ids = torch.cat(train_input_ids, dim=0)
        return train_input_ids

    def batch_encode_plus(self, sentences, padding='max_length', verbose=True):
        train_input_ids = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                max_length=self.max_seq_length,
                padding=padding,
                #pad_to_max_length=True,
                truncation=True
            )
        return train_input_ids

    def add_pooling_layer(self):
        if not hasattr(self, 'embedding_pooling'):# and self.task == "reconstruction":
            self.embedding_pooling = Pooling(self.get_word_embedding_dimension(),
                                     pooling_mode_mean_tokens=True,
                                     pooling_mode_cls_token=False,
                                     pooling_mode_max_tokens=False)
