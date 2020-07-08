import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import logging
from transformers import modeling_bart


class EncDecModel(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None ):
        super(EncDecModel, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.config = model.config
        self.config_class = model.config_class
        self.device = model.device
        self.dtype = model.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)


        self.output_attentions = True
        self.output_hidden_states = True
        self.config.output_attentions = True
        self.config.output_hidden_states = True
        self.encoder.output_attentions = True
        self.encoder.output_hidden_states = True
        self.decoder.output_attentions = True
        self.decoder.output_hidden_states = True

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        decoder_input_ids=None
        if 'decoder_input_ids' not in features:
            use_cache = False
        else:
            decoder_input_ids= features['decoder_input_ids']

        decoder_attention_mask= features['decoder_attention_mask'] if 'decoder_attention_mask' in features else None
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        decoder_cached_states = features['decoder_cached_states'] if 'decoder_cached_states' in features else None

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = modeling_bart._prepare_bart_decoder_inputs(
                self.config,
                features['input_ids'],
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if 'encoder_outputs' not in features:
            encoder_outputs = self.encoder(
                input_ids=features['input_ids'],
                attention_mask=features['attention_mask'],
            )

        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            features['attention_mask'],
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = modeling_bart._filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = modeling_bart._filter_out_falsey_values(encoder_outputs)

        return decoder_outputs + encoder_outputs




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
        return EncDecModel(model_name_or_path=input_path, **config)

    def encode(self, sentences):
        logging.info("Trainer - encoding training data")
        train_input_ids = []
        for text in tqdm(sentences):
            input_ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_tensors='pt'
            )
            train_input_ids.append(input_ids)
        train_input_ids = torch.cat(train_input_ids, dim=0)
        return train_input_ids

    def encodeSentence(self,sentence):
        logging.info("Trainer - encoding sentence")
        train_input_ids = []
        input_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_tensors='pt'
            )
        return input_ids

