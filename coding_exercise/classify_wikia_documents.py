"""
Author: @Isaac Caswell
May 2014
This implementation can be extended to discriminate between an arbitrary amount of genres.

Evaluation of bug: regardless of the query, the similarity vector over the training data that is returned
is almost identical.  For instance, the training document in position one is almost always ranked 206
in terms of similarity, and the last document is given the rank 134.  This doesn't a;ways happen, but 
happens far too often to be chance.

Ways to improve:
)Parse text better
-->only include words with a tfidf above a certain threshhold
)Use a different distance metric
-->Heilinger distance instead of cosine distance?
)Use a different model
-->LSI
-->Markov chain

naming conventions: 
text - the (processed) representation of a document as a string or a list of strings
tests -  a collection of texts
doc - any abstract numerical representation of a document (i.e. counts and ids)
corpus - a collection of docs
"""
from scipy import spatial
import bz2
import gensim
import itertools
import logging
import numpy as np
import pickle #for loading/saving the training labels
import simple_text_parser as stp #(I wrote this module)
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#Variables to control overall program flow
EXTRACT_TEXT = 0 # scrape texts off Wikia, parse and save to file
LEARN_MODEL = 1 #Learn the LDA model
TEST_MODEL = 1 #Evaluate the model on the held-out test set
print "run on %s: \n\tEXTRACT_TEXT = %s\n\tLEARN_MODEL = %s\n\tTEST_MODEL = %s" \
    %(time.asctime(), EXTRACT_TEXT, LEARN_MODEL, TEST_MODEL)

#Global constants
HOLD_OUT_PCT = .3
N_ITER = 50000 
N_TOPICS = 2
UNIVERSES = ['dc', 'marvel']

#Filenames, so that the learning and testing can be done separately
#central control of file stem labelling allows for easy restructuring of file systems/naming conventions, if desired
common_file_stem = '../files/'
full_corp_file_stem = common_file_stem + "+".join(UNIVERSES)
CORPUS_FILENAME = full_corp_file_stem +'_corpus.mm'
TRAIN_LABELS_FILENAME = full_corp_file_stem +'_labels' 
DICT_FILENAME = full_corp_file_stem +'_corpus.dict'
LDA_FILENAME = full_corp_file_stem + '_lda_%s_iter_K=%s.lda'%(N_ITER, N_TOPICS)
INDEX_FILENAME = full_corp_file_stem +'.index'

#++++++++++++
#CorpusGenerator: create and save corpus and dictionary.  Used in both TEST and TRAIN.  
#I didn't really know where better to put this definition.
#++++++++++++
class CorpusGenerator():
    """
    A generator to load a corpus from a file and convert it into the bow format.  
    The file should have each document on a separate line, with each token separated with whitespace.
    I use a a generator (lazy iterator) because it is better for dealing with massive amounts of data, 
    so as not to exhaust my poor computer's RAM.
    """
    def __init__(self, dictionary, file_stem):
        self._dictionary = dictionary
        self._file_stem = file_stem
        #the lines_read variable allows one to determine the line count
        #of a file after the generator has been used
        self.lines_read = 0
    
    def __iter__(self):
        for line in open(self._file_stem):
            # assume there's one document per line, tokens separated by whitespace
            self.lines_read +=1
            yield self._dictionary.doc2bow(line.split())
        

if EXTRACT_TEXT:
    """
    get texts from Wikia and save corpi from the various universes in files 
    for training and testing.  Makes seperate testing and training files for each universe,
    such as dc_texts_test.txt
    """
    import extract_text as et #(I wrote this module)
    for universe in UNIVERSES:
        et.parse_and_save_corpus_to_file(common_file_stem + universe +'_texts', universe, HOLD_OUT_PCT)#, debug = {"percent_articles_to_consider":.01})



if LEARN_MODEL:
    texts_generators = ((stp.tokenize_string_to_list(line) for line in \
                         open(common_file_stem + universe +'_texts_train')) for universe in UNIVERSES)
    all_texts = itertools.chain(*texts_generators)
    dictionary = gensim.corpora.Dictionary(all_texts)
    
    #Non generator method:
    ##corpus = [dictionary.doc2bow(doc) for doc in corpus]
    #corpus = CorpusGenerator(dictionary, corp_name)#'../files/toy_texts', dictionary)
    
    corpus_generators = [CorpusGenerator(dictionary, common_file_stem + universe + '_texts_train')\
                          for universe in UNIVERSES]
    corpus = itertools.chain(*corpus_generators)
    
    #save corpora to file
    gensim.corpora.MmCorpus.serialize(CORPUS_FILENAME, corpus)

    #create the vector of document labels
    #IMPORTANT NOTE: the cg.lines_read trick ONLY WORKS if the generator has already been used.
    #This, the train labels must be gotten after (in this case) serializing the corpus to a file
    sub_corpus_sizes = [cg.lines_read for cg in corpus_generators]
    train_labels = []
    for i in range(len(sub_corpus_sizes)):
        for _ in range(sub_corpus_sizes[i]):
            train_labels.append(UNIVERSES[i])
    
    #save dictionary to file
    dictionary.save(DICT_FILENAME)
    
    #save labels to file
    pickle.dump(train_labels, open(TRAIN_LABELS_FILENAME, 'wb'))
    
    #would it be faster to do it as stochastic batch LDA? Or does it already do this internally? 
    lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=N_TOPICS, iterations=N_ITER)
    lda.save(LDA_FILENAME)

    ##TODO: use similarity.Similarity for ease on RAM
    ##Note: unsure why the syntax gensim.similarities.MatrixSimilarity(corpus) doesn't work
    index = gensim.similarities.MatrixSimilarity(gensim.corpora.MmCorpus(CORPUS_FILENAME)) 
    index.save(INDEX_FILENAME)


if TEST_MODEL:
    """
    Now, assuming that the model has already been trained and everything is saved into files,
    load up the models we learned, and test classification with the test sets.
    """
    train_labels = np.array(pickle.load(open(TRAIN_LABELS_FILENAME, 'rb')))
    dictionary = gensim.corpora.Dictionary.load(DICT_FILENAME)
    corpus = gensim.corpora.MmCorpus(CORPUS_FILENAME)
    lda = gensim.models.ldamodel.LdaModel.load(LDA_FILENAME)
    index = gensim.similarities.MatrixSimilarity.load(INDEX_FILENAME)
        
    #===============================================================================
    # load test set into a generator
    #===============================================================================
    test_set_generators = [CorpusGenerator(dictionary, common_file_stem + universe + '_texts_test')\
                          for universe in UNIVERSES]
    
    
    #=============================================================================#
    # Functions to predict the category of a document given an trained LDA model, 
    #=============================================================================#
    def predict_universe_nearest_neighbor(query, train_labels, lda, index):
        """
        Use LDA to find the closest document from the training corpus to the query 
        doc, and return the class label associated therewith.  Simplistic, but let's 
        see how it works?
        """
        #convert the query to LDA space:
        query_lda = lda[query]
        sims = index[query_lda]
        #     cosine_distances = [spatial.distance.cosine(query, train_doc_topics)\
        #                      for train_doc_topics in sims]
        
        return train_labels[np.argmax(sims)]
    
    def predict_universe_avg_dist(query, train_labels, lda, index):
        """
        Use LDA to find a vector of the closest documents from the training corpus to the query 
        doc.  return the category whose average distance is the lowest.
        """
        query_lda = lda[query]
        sims = index[query_lda]
        #partitioned_space: a list of lists, where each sublist corresponds to a single universe, and 
        #each element therein is the distance of said doc to the query doc.
        partitioned_space = [sims[np.where(train_labels == universe)] for universe in UNIVERSES]
        lg_avg_dists = [(np.log(partition.sum()) - np.log(sims.shape[0])) for partition in partitioned_space]
        return UNIVERSES[np.argmax(lg_avg_dists)]
    
    def predict_universe_weighted_avg_dist(query, train_labels, lda, index, alpha = 1.0):
        """
        Use LDA to find a vector of the closest documents from the training corpus to the query 
        doc.  Weight each doc by alpha, so closer docs have more influence on the average (if alpha <1) 
        or less influence on the average (if alpha > 1)
        
        Note: train_labels is a numpy array.  IF it is given as a list, program will appear to function but will fail.
        """
        query_lda = lda[query]
        sims = index[query_lda]
        ordinals = np.argsort(sims)
        #Exponential decay
        sims = np.array([sims[i]*(alpha**ordinals[i]) for i in range(ordinals.shape[0])])
        #partitioned_space: a list of arrays, where each array corresponds to a single universe, and 
        #each element therein is the distance of said doc to the query doc.
        partitioned_space = [sims[np.where(train_labels == universe)] for universe in UNIVERSES]
        avg_dists = [(np.log(partition.sum()) - np.log(partition.shape[0])) for partition in partitioned_space]
        return UNIVERSES[np.argmax(avg_dists)]
    
    
    #=============================================================================#
    # Evaluate accuracy of model
    #=============================================================================#
    
    total = 0
    total_correct = 0
    test_universe_idx = 0
    for test_corpus in test_set_generators:
        for doc in test_corpus:
            pred = predict_universe_nearest_neighbor(doc, train_labels, lda, index)
            #pred = predict_universe_avg_dist(doc, train_labels, lda, index)
            #pred = predict_universe_weighted_avg_dist(doc, train_labels, lda, index, alpha = .9)
            print "%s %s"%(pred, UNIVERSES[test_universe_idx])
            if pred == UNIVERSES[test_universe_idx]:
                total_correct +=1
            total +=1
        test_universe_idx +=1
    
    accuracy = 1.0*total_correct/total
    print "accuracy: %s"%accuracy

