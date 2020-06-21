from torch.utils.data import Dataset
import torch
import logging
import gzip
import os
import random

from tqdm import tqdm

from . import BaseDataset
import urllib.request



class ParallelSentencesDataset(BaseDataset):
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings


    def __init__(self, student_model: SentenceTransformer, teacher_model):
        Parallel sentences dataset reader to train student model given a teacher model
        :param student_model: Student sentence embedding model that should be trained
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
    """


    def __init__(self,opt,model):
        """
        Parallel sentences dataset reader to train student model given a teacher model
        :param opt: options used to create and read the dataset
        """
        BaseDataset.__init__(self, opt)
        self.model = model
        self.filepaths = ["TED2013-en-de.txt.gz", "STS2017.en-de.txt.gz", "xnli-en-de.txt.gz"]
        self.datasets = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.server = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"
        self.model=model

        for dataset in self.filepaths:
            print("Download", dataset)
            url = self.server+dataset
            dataset_path = os.path.join(self.root, dataset)
            urllib.request.urlretrieve(url, dataset_path)

        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory



    def load_data(self):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more that one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :return:
        """
        filepath = os.path.join(self.root, self.filepaths[0])
        weight = self.opt.param_weight
        max_sentences = self.opt.max_sentences
        if max_sentences == 0:
            max_sentences = None
        max_sentence_length = self.opt.max_sentence_length

        sentences_map = {}
        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                sentence_lengths = [len(sent) for sent in sentences]
                if max(sentence_lengths) > max_sentence_length:
                    continue

                eng_sentence = sentences[0]


                if eng_sentence not in sentences_map:
                    sentences_map[eng_sentence] = set()

                for sent in sentences:
                    if sent != eng_sentence:
                        sentences_map[eng_sentence].add(sent)

                count += 1
                if max_sentences is not None and count >= max_sentences:
                    break

        eng_sentences = list(sentences_map.keys())

        logging.info("Create sentence embeddings for " + os.path.basename(filepath))
        #encodings= self.model.netG_A.module.tokenize(eng_sentences)
        if self.opt.model == 'cycle_gan':
            eng_encodings = self.model.netG_A.module.encode(
                eng_sentences)  # , batch_size=32, show_progress_bar=True), dtype=torch.float)
        elif self.opt.model == 'gan':
            eng_encodings = self.model.netref.module.encode(eng_sentences)  # , batch_size=32, show_progress_bar=True), dtype=torch.float)

        self.dir_AB = os.path.join(self.opt.dataroot, self.opt.phase)  # get the image directory

        data = []
        for idx in tqdm(range(len(eng_sentences))):
            eng_key = eng_sentences[idx]
            embedding = eng_encodings[idx]
            for sent in sentences_map[eng_key]:
                if self.opt.model == 'cycle_gan':
                    data.append([self.model.netG_B.module.encodeSentence(sent), embedding])
                elif self.opt.model == 'gan':
                    data.append([self.model.netref.module.encodeSentence(sent), embedding])

        dataset_id = len(self.datasets)
        self.datasets.append(data)
        self.dataset_indices.extend([dataset_id] * weight)


    def __len__(self):
        return max([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        if len(self.copy_dataset_indices) == 0:
            self.copy_dataset_indices = self.dataset_indices.copy()
            random.shuffle(self.copy_dataset_indices)

        dataset_idx = self.copy_dataset_indices.pop()

        A = self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])][0][0]

        B = self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])][1]



        return {'A': A, 'B': B, 'A_paths': self.filepaths[dataset_idx], 'B_paths': self.filepaths[dataset_idx]}
        return self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])]
