"""
This script tests the approach on the BUCC 2018 shared task on finding parallel sentences:
https://comparable.limsi.fr/bucc2018/bucc2018-task.html

You can download the necessary files from there.

We have used it in our paper (https://arxiv.org/pdf/2004.09813.pdf) in Section 4.2 to evaluate different multilingual models.

This script requires that you have FAISS installed:
https://github.com/facebookresearch/faiss
"""
import argparse

from models import EncDecModel
from collections import defaultdict
import os
import pickle
from sklearn.decomposition import PCA
import torch
from bitext_mining_utils import *
import numpy as np



parser = argparse.ArgumentParser(description='')
parser.add_argument('--language', required=True, help='Language compared to English')
parser.add_argument('--models_dir', type=str, default='../../checkpoints/translation',
                    help='Root directory of saved models')
parser.add_argument('--models_prefix', type=str, default='latest',
                    help='Prefix in model name')

args = parser.parse_args()


#Model we want to use for bitext mining. LaBSE achieves state-of-the-art performance
prefix = args.models_prefix
model_dir = args.models_dir
modelA_name = prefix+'_net_G_AB'
modelB_name = prefix+'_net_G_BA'
modelA = EncDecModel(os.path.join(model_dir, modelA_name))
modelB = EncDecModel(os.path.join(model_dir, modelB_name))

language = args.language
#Intput files for BUCC2018 shared task
source_file = "bucc2018/"+language+"-en/"+language+"-en.training."+language
target_file = "bucc2018/"+language+"-en/"+language+"-en.training.en"
labels_file = "bucc2018/"+language+"-en/"+language+"-en.training.gold"



# We base the scoring on k nearest neighbors for each element
knn_neighbors = 4

# Min score for text pairs. Note, score can be larger than 1
min_threshold = 1

#Do we want to use exact search of approximate nearest neighbor search (ANN)
#Exact search: Slower, but we don't miss any parallel sentences
#ANN: Faster, but the recall will be lower
use_ann_search = False

#Number of clusters for ANN. Optimal number depends on dataset size
ann_num_clusters = 32768

#How many cluster to explorer for search. Higher number = better recall, slower
ann_num_cluster_probe = 5

#To save memory, we can use PCA to reduce the dimensionality from 768 to for example 128 dimensions
#The encoded embeddings will hence require 6 times less memory. However, we observe a small drop in performance.
use_pca = False
pca_dimensions = 128

#We store the embeddings on disc, so that they can later be loaded from disc
source_embedding_file = '{}_{}_{}.emb'.format(modelA_name, os.path.basename(source_file), pca_dimensions if use_pca else modelA.get_sentence_embedding_dimension())
target_embedding_file = '{}_{}_{}.emb'.format(modelB_name, os.path.basename(target_file), pca_dimensions if use_pca else modelB.get_sentence_embedding_dimension())


#Use PCA to reduce the dimensionality of the sentence embedding model

print("Read source file")
source = {}
with open(source_file, encoding='utf8') as fIn:
    for line in fIn:
        id, sentence = line.strip().split("\t", maxsplit=1)
        source[id] = sentence

print("Read target file")
target = {}
with open(target_file, encoding='utf8') as fIn:
    for line in fIn:
        id, sentence = line.strip().split("\t", maxsplit=1)
        target[id] = sentence

labels = defaultdict(lambda: defaultdict(bool))
num_total_parallel = 0
with open(labels_file) as fIn:
    for line in fIn:
        src_id, trg_id = line.strip().split("\t")
        if src_id in source and trg_id in target:
            labels[src_id][trg_id] = True
            labels[trg_id][src_id] = True
            num_total_parallel += 1

print("Source Sentences:", len(source))
print("Target Sentences:", len(target))
print("Num Parallel:", num_total_parallel)

### Encode source sentences
source_ids = list(source.keys())
source_sentences = [source[id] for id in source_ids]

if not os.path.exists(source_embedding_file):
    print("Encode source sentences")
    source_embeddings = modelA.encode(source_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open(source_embedding_file, 'wb') as fOut:
        pickle.dump(source_embeddings, fOut)
else:
    with open(source_embedding_file, 'rb') as fIn:
        source_embeddings = pickle.load(fIn)

### Encode target sentences
target_ids = list(target.keys())
target_sentences = [target[id] for id in target_ids]

if not os.path.exists(target_embedding_file):
    print("Encode target sentences")
    target_embeddings = modelB.encode(target_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open(target_embedding_file, 'wb') as fOut:
        pickle.dump(target_embeddings, fOut)
else:
    with open(target_embedding_file, 'rb') as fIn:
        target_embeddings = pickle.load(fIn)

##### Now we start to search for parallel (translated) sentences

# Normalize embeddings
x = source_embeddings
y = target_embeddings

print("Shape Source:", x.shape)
print("Shape Target:", y.shape)

x = x / np.linalg.norm(x, axis=1, keepdims=True)
y = y / np.linalg.norm(y, axis=1, keepdims=True)

# Perform kNN in both directions
x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
x2y_mean = x2y_sim.mean(axis=1)

y2x_sim, y2x_ind = kNN(y, x, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
y2x_mean = y2x_sim.mean(axis=1)

# Compute forward and backward scores
margin = lambda a, b: a / b
fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

indices = np.stack([np.concatenate([np.arange(x.shape[0]), bwd_best]), np.concatenate([fwd_best, np.arange(y.shape[0])])], axis=1)
scores = np.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
seen_src, seen_trg = set(), set()

#Extact list of parallel sentences
bitext_list = []
for i in np.argsort(-scores):
    src_ind, trg_ind = indices[i]
    src_ind = int(src_ind)
    trg_ind = int(trg_ind)

    if scores[i] < min_threshold:
        break

    if src_ind not in seen_src and trg_ind not in seen_trg:
        seen_src.add(src_ind)
        seen_trg.add(trg_ind)
        bitext_list.append([scores[i], source_ids[src_ind], target_ids[trg_ind]])


# Measure Performance by computing the threshold
# that leads to the best F1 score performance
bitext_list = sorted(bitext_list, key=lambda x: x[0], reverse=True)

n_extract = n_correct = 0.0
threshold = 0.0
best_f1 = best_recall = best_precision = 0.0
average_precision = 0.0

for idx in range(len(bitext_list)):
    score, id1, id2 = bitext_list[idx]
    n_extract += 1
    if labels[id1][id2] or labels[id2][id1]:
        n_correct += 1
        precision = (float)(n_correct) / n_extract
        recall = (float)(n_correct) / num_total_parallel
        f1 = 2.0 * precision * recall / (precision + recall)
        average_precision += precision
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            threshold = (bitext_list[idx][0] + bitext_list[min(idx + 1, len(bitext_list)-1)][0]) / 2

print("Best Threshold:", threshold)
print("Recall:", best_recall)
print("Precision:", best_precision)
print("F1:", best_f1)