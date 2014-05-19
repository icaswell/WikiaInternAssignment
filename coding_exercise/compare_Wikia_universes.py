"""
"""

import extract_text
from compute_tf_idfs import tf_idf_for_corpus
import numpy as np
import scipy.sparse as spr

universes = ["marvel", "dc"]
corpus = extract_text.get_corpus(universes[1], debug = {"percent_articles_to_consider":.2})

#tfidf_data['tf_dfs'] is a WxD matrix of tf_idfs, 
# where W is the vocabulary size and D the amount of documents.
tf_idf_data = tf_idf_for_corpus(corpus)

#tfs: logarithmically scaled term frequencies
tfs = tf_idf_data['tfs']
words = np.array(tf_idf_data['words'])
corpus_wide_tfs = tfs.sum(1)
corpus_wide_tfs = np.squeeze(np.asarray(corpus_wide_tfs))

#vocab maps words to integer ids
print corpus_wide_tfs
argstd_tfs = np.argsort(corpus_wide_tfs)


#this number was determined by looking at a plot of the tf-idf scores, and is highly approximate.
threshhold = 60000 
words = words[argstd_tf_idfs[threshhold:]]
vocab = vocab_from_words(words)
W = len(vocab)
wordids, wordcts = parse_doc_list(corpus, vocab)
counts = make_sparse_ct_matrices(wordids, wordcts, W)