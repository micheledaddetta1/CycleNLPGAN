"""
This script downloads the TED2020 corpus (https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md)
 and create parallel sentences tsv files that can be used to extend existent sentence embedding models to new languages.

The TED2020 corpus is a crawl of transcripts from TED and TEDx talks, which are translated to 100+ languages.

The TED2020 corpus is downloaded automatically. Otherwise, it can be found here:
https://sbert.net/datasets/ted2020.tsv.gz

The training procedure can be found in the files make_multilingual.py and make_multilingual_sys.py.

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""
import os
from util import util
import gzip
import csv
from tqdm.autonotebook import tqdm


source_languages = list(['en'])                                  #Languages our (monolingual) teacher model understands
target_languages = list(['de', 'es', 'it', 'fr', 'ru', 'zh'])    #New languages we want to extend to


eval_sentences = 1000         #Number of sentences we want to use for evalelopment
download_url = "https://sbert.net/datasets/ted2020.tsv.gz"
ted2020_path = "./ted2020.tsv.gz" #Path of the TED2020.tsv.gz file.
parallel_sentences_folder = "./"




os.makedirs(os.path.dirname(ted2020_path), exist_ok=True)
if not os.path.exists(ted2020_path):
    print("ted2020.tsv.gz does not exists. Try to download from server")
    util.http_get(download_url, ted2020_path)



os.makedirs(parallel_sentences_folder, exist_ok=True)
train_files = []
eval_files = []
files_to_create = []
for source_lang in source_languages:
    for target_lang in target_languages:
        output_filename_train = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-train.tsv.gz".format(source_lang, target_lang))
        output_filename_eval = os.path.join(parallel_sentences_folder, "TED2020-{}-{}-eval.tsv.gz".format(source_lang, target_lang))
        train_files.append(output_filename_train)
        eval_files.append(output_filename_eval)
        if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_eval):
            files_to_create.append({'src_lang': source_lang, 'trg_lang': target_lang,
                                    'fTrain': gzip.open(output_filename_train, 'wt', encoding='utf8'),
                                    'feval': gzip.open(output_filename_eval, 'wt', encoding='utf8'),
                                    'evalCount': 0
                                    })

if len(files_to_create) > 0:
    print("Parallel sentences files {} do not exist. Create these files now".format(", ".join(map(lambda x: x['src_lang']+"-"+x['trg_lang'], files_to_create))))
    with gzip.open(ted2020_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in tqdm(reader, desc="Sentences"):
            for outfile in files_to_create:
                src_text = line[outfile['src_lang']].strip()
                trg_text = line[outfile['trg_lang']].strip()

                if src_text != "" and trg_text != "":
                    if outfile['evalCount'] < eval_sentences:
                        outfile['evalCount'] += 1
                        fOut = outfile['feval']
                    else:
                        fOut = outfile['fTrain']

                    fOut.write("{}\t{}\n".format(src_text, trg_text))

    for outfile in files_to_create:
        outfile['fTrain'].close()
        outfile['feval'].close()


print("---DONE---")