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

    def forward(self, sentences, target_sentences=None, partial_value=False):

        embeddings = self.batch_encode_plus(sentences, padding=True, verbose=False)
        embeddings = embeddings.to(self.model.device)
        if self.task == "translation":
            if target_sentences is not None:
                decoder_input_ids = self.batch_encode_plus(target_sentences,  padding=True, verbose=False).input_ids.to(self.model.device)  # Batch size 1
                outputs = self.model(input_ids=embeddings.input_ids, labels=decoder_input_ids, return_dict=True)
            else:
                outputs = self.model(input_ids=embeddings.input_ids, return_dict=True)
            output_sentences = self.model.generate(**embeddings)
            output_sentences = self.decode(output_sentences)
        else:
            output_sentences = self.generate(sentences)
            output_sentences = self.decode(output_sentences)


        if partial_value:

            sentence_embedding = torch.zeros([len(sentences), self.get_word_embedding_dimension()], dtype=torch.float32).to(self.model.device)
            for i in range(len(sentences)):
                a = outputs.encoder_last_hidden_state[i]
                sentence_embedding[i] = self.embedding_pooling(outputs.encoder_last_hidden_state[i])["sentence_embedding"]
            if target_sentences is not None:
                return output_sentences, sentence_embedding, outputs.loss
            else:
                return output_sentences, sentence_embedding
        else:
            del embeddings

            if target_sentences is not None:
                return output_sentences, outputs.loss
            else:
                return output_sentences

    def get_word_embedding_dimension(self) -> int:
        return self.config.hidden_size

    def get_sentence_embedding_dimension(self) -> int:
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
    def load(input_path: str, task, freeze_encoder):
        #with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
        #    config = json.load(fIn)

        return EncDecModel(model_name_or_path=input_path, task=task, freeze_encoder=freeze_encoder) #, **config)


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

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding', convert_to_numpy: bool = True) -> List[np.ndarray]:
        """
        Computes sentence embeddings

        :param sentences:
           the sentences to embed
        :param batch_size:
           the batch size used for the computation
        :param show_progress_bar:
            Output a progress bar when encode sentences
        :param output_value:
            Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings
            to get wordpiece token embeddings.
        :param convert_to_numpy:
            If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :return:
           Depending on convert_to_numpy, either a list of numpy vectors or a list of pytorch tensors
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                        logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                # features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)
                features[feature_name] = torch.cat(features[feature_name]).to(self.model.device)

            with torch.no_grad():
                _, out_features = self.forward(features, partial_value=True)
                embeddings = out_features

                if output_value == 'token_embeddings':
                    # Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                if convert_to_numpy:
                    embeddings = embeddings.to('cpu').numpy()

                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings


    def batch_encode_plus(self, sentences, padding='max_length', verbose=True):
        train_input_ids = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding=padding,
                    #pad_to_max_length=True,
                    truncation=True,
                )
        return train_input_ids

    def add_pooling_layer(self):
        if not hasattr(self, 'embedding_pooling'):# and self.task == "reconstruction":
            self.embedding_pooling = Pooling(self.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True,
                                 pooling_mode_cls_token=False,
                                 pooling_mode_max_tokens=False)

