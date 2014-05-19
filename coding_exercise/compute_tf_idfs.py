from collections import Counter
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pylab import *
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.io as sio
import scipy.sparse as spr
import string
import sys
import time



def parse_doc_list(docs, vocab):
    """
    Function lifted from Matthew D. Hoffman, 2010
    
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    print 'parsing doc lists....'
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    
    for d in range(0, D):
        print '\tparsing doc %i...' % d
        #docs[d] = tokenize_doc(docs[d])  # , go_words = vocab.keys())
        ddict = dict()
        for word in docs[d]:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())
          
 
        #=======================================================================
        # sc = time.clock()
        # c = Counter(docs[d])
        # docwords = c.keys()
        # wordids.append([vocab[w] for w in docwords])
        # wordcts.append([c[w] for w in docwords])
        # print time.clock() - sc
        #=======================================================================
        
    return (wordids, wordcts)


def make_sparse_ct_matrices(wordids, wordcts, W):
    counts = spr.csc_matrix((W, len(wordids)))
    for doc in range(len(wordids)):
        for word in range(len(wordids[doc])):
            counts[wordids[doc][word], doc] = wordcts[doc][word]
        print 'sparsified doc %i' % doc
    return counts 


def words_from_corpus(corpus):
    """
    corpus is a list of docs, where each doc is a list of strings.  Stop words have been removed.
    """
    flattened_corpus = reduce(lambda a, b: a  + b, corpus)
    flattened_corpus = set(flattened_corpus)
    words = sorted(list(set(flattened_corpus)))
    return words


def get_idfs(tfs):
    W, D = tfs.shape
    doc_occurrence = Counter(tfs.nonzero()[0])
    doc_occurrence = np.array([doc_occurrence[row] for row in doc_occurrence])  # convert to a list bzw array
    idfs = np.log(D / doc_occurrence)
    idfs = idfs.reshape(idfs.shape[0], 1)
    return idfs
    
    
    
def vocab_from_words(words):
    '''
    words is an array of strings; vocab is a dict of strings to their indices.
    '''
    vocab = dict()
    for word in words:
        word = word.lower()
        #word = re.sub(r'[^a-z]', '', word)  # redundant for most places this function is called.
        vocab[word] = len(vocab)
    return vocab

    
def make_tfs(corpus, trial_id):
    words = words_from_corpus(corpus)
    vocab = vocab_from_words(words)
    W = len(vocab)
    
#     if type == 'gesamt':
#         print 'flattening corpus...'
#         flattened_corpus = tokenize_doc(docs, stop_words=stop_words, corpus=1)
#         words = sorted(list(set(flattened_corpus)))  # uses about .22 secs for the country dataset
#         vocab = vocab_from_words(words)
#         W = len(vocab)
#         tfs = Counter(flattened_corpus)
#         tfs = [tfs[word] for word in words]  # thus: sorted in same order as words.
#         d = {'tfs': tfs, 'words': words}
        
    #if type == 'einzeln':
    wordids, wordcts = parse_doc_list(corpus, vocab)
    #TODO: why not just use .tosparse() ?  Check once other things are working
    tfs = make_sparse_ct_matrices(wordids, wordcts, W)
    d = {'tfs': tfs, 'words': words}
    return d
    #sio.savemat('tfs_%s.mat'%trial_id, d)



def tf_idf_for_corpus(corpus, trial_id = ""):
    '''Calculates tfidfs for all words occurring within a document.  
    Does not remove stop words
    '''

    tfs = make_tfs(corpus[:], trial_id)
       
    #tfs = sio.loadmat('tfs_%s.mat'%trial_id)
    words = tfs['words']
    vocab = vocab_from_words(words)
    tfs = tfs['tfs']
    W, D = tfs.shape
    idfs = get_idfs(tfs)
    #logarithmically scaled frequency
    tfs = log(1 + tfs.todense())#tfs.log1p()
    
    tf_idfs = np.multiply(tfs, np.tile(idfs, D))#tfs.multiply(np.tile(idfs, D))#np.multiply(tfs, np.tile(idfs, D)) 
    #tf_idfs = tf_idfs.sum(1)
    
    d = {'tf_idfs': tf_idfs, 'tfs': tfs, 'words': words, 'vocab': vocab}
    #sio.savemat('tf_idfs_%s.mat'%trial_id, d)
    
    #tf_idfs = sio.loadmat('tf_idfs_%s.mat'%trial_id)
    #words = tf_idfs['words']
    #tf_idfs = tf_idfs['tf_idfs']
    
    return d

    """below: for plotting tf_idfs"""
    #===========================================================================
    # tf_idfs_for_plotting = np.squeeze(sorted(tf_idfs))
    # print np.array(tf_idfs_for_plotting).shape
    # plt.plot(np.array(tf_idfs_for_plotting))
    # plt.show()
    #===========================================================================
    
    """Below: processing for LDA  Of yet unused, but might be incorporated into functionality later."""
    #===========================================================================
    # argstd_tf_idfs = np.argsort(tf_idfs)
    # #this number was determined by looking at a plot of the tf-idf scores, and is highly approximate.
    # threshhold = 60000 
    # words = words[argstd_tf_idfs[threshhold:]]
    # vocab = vocab_from_words(words)
    # W = len(vocab)
    # wordids, wordcts = parse_doc_list(corpus, vocab)
    # counts = make_sparse_ct_matrices(wordids, wordcts, W)
    # description = 'tf-idfs calculated; %s words with highest tf-idf kept.'%threshhold
    # return {'counts': counts, 'words': words, 'description': description}
    # 
    #===========================================================================
    
    
    #sio.savemat('../../data/wiki_countries_3', data)
    
