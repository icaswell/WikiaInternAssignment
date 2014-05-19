"""
for testing my modules
"""


#=======================================
#Gensim LDA
#=======================================


import logging, gensim, bz2
from scipy import spatial
import itertools
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import simple_text_parser as stp
import numpy as np
#central control of file stem labelling allows for easy restructuring of file systems/naming conventions, if desired

#
LEARN_MODEL = 0
N_ITER = 5000 #TODO change to something reasonable like 5000
N_TOPICS = 5
UNIVERSES = ['dc', 'marvel']

common_file_stem = '../files/'
full_corp_file_stem = common_file_stem + "+".join(UNIVERSES)
CORPUS_FILENAME = full_corp_file_stem +'_corpus.mm'
TRAIN_LABELS_FILENAME = full_corp_file_stem +'_labels' 
DICT_FILENAME = full_corp_file_stem +'_corpus.dict'
LDA_FILENAME = full_corp_file_stem + '_lda_%s_iter_K=%s'%(N_ITER, N_TOPICS)
INDEX_FILENAME = full_corp_file_stem +'.index'




#++++++++++++
#create and save corpus and dictionary.
#++++++++++++
class CorpusGenerator():
    """
    using a generator (lazy iterator) is better for dealing with massive amounts of data, 
    so as not to exhaust my poor computer's RAM
    """
    def __init__(self, dictionary, file_stem):
        self._dictionary = dictionary
        self._file_stem = file_stem
        self.lines_read = 0
    
    def __iter__(self):
        for line in open(self._file_stem):
            # assume there's one document per line, tokens separated by whitespace
            self.lines_read +=1
            yield self._dictionary.doc2bow(line.split())
        

if LEARN_MODEL:
    texts_generators = ((stp.tokenize_string_to_list(line) for line in \
                         open(common_file_stem + universe +'_texts_train')) for universe in UNIVERSES)
    all_texts = itertools.chain(*texts_generators)
    dictionary = gensim.corpora.Dictionary(all_texts)
    

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
    lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10, iterations=N_ITER)
    lda.save(LDA_FILENAME)
    #lda.print_topics(5)

    ##TODO: use similarity.Similarity for ease on RAM
    ##Note: unsure why gensim.similarities.MatrixSimilarity(corpus) doesn't work
    index = gensim.similarities.MatrixSimilarity(gensim.corpora.MmCorpus(CORPUS_FILENAME)) 
    index.save(INDEX_FILENAME)

#===
#Now, assuming everything is saved into files, load them back up.
#Why I decided to save everything to files I'm not sure.  I think it was because this way, 
#it's easier to generalize to massive projects.

train_labels = pickle.load(open(TRAIN_LABELS_FILENAME, 'rb'))
dictionary = gensim.corpora.Dictionary.load(DICT_FILENAME)
corpus = gensim.corpora.MmCorpus(CORPUS_FILENAME)
lda = gensim.models.ldamodel.LdaModel.load(LDA_FILENAME)
index = gensim.similarities.MatrixSimilarity.load(INDEX_FILENAME)




#===============================================================================
# TEST MODEL
#===============================================================================


#===============================================================================
# load test set
#===============================================================================
test_set_generators = [CorpusGenerator(dictionary, common_file_stem + universe + '_texts_test')\
                      for universe in UNIVERSES]

#===============================================================================
# class chain_and_iterate(object):
#     """
#     I feel that this is a bad solution but I am tired
#     """
#     def __iter__(self):
#         for tsg in test_set_generators:
#             for doc in tsg:
#                 yield doc
# #test_set = itertools.chain(*test_set_generators)
# test_set = chain_and_iterate()
# 
# #create the vector of document labels
# sub_corpus_sizes = [cg.lines_read for cg in test_set_generators]
# test_labels = []
# for i in range(len(sub_corpus_sizes)):
#     for _ in range(sub_corpus_sizes[i]):
#         test_labels.append(UNIVERSES[i])
#===============================================================================


#=============================================================================#
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
    #TODO: this ain't right
#     cosine_distances = [spatial.distance.cosine(query, train_doc_topics)\
#                      for train_doc_topics in sims]
    return train_labels[np.argmax(sims)]

def predict_universe_avg_dist(query, train_labels, lda, index):
    query_lda = lda[query]
    
    sims = index[query_lda]
    #std_order = np.argsort(sims)
    partitioned_space = [sims[np.where(train_labels == universe)] for universe in UNIVERSES]
    avg_dists = [np.mean(partition) for partition in partitioned_space]
    return train_labels[np.argmax(avg_dists)]


#=============================================================================#
#=============================================================================#

total = 0
total_correct = 0
test_universe = 0
for test_corpus in test_set_generators:
    for doc in test_corpus:
        #pred = predict_universe_nearest_neighbor(doc, train_labels, lda, index)
        
        pred = predict_universe_avg_dist(doc, train_labels, lda, index)
        if pred == UNIVERSES[test_universe]:
            total_correct +=1
        total +=1
        print pred + UNIVERSES[test_universe]
    test_universe +=1

accuracy = 1.0*total_correct/total
print "accuracy: %s"%accuracy

#=============================================================================#
#=============================================================================#





#===============================================================================
# EXEMPLAR METHOD
#===============================================================================


#==========================================
#Markov Chain
#========================================

#===============================================================================
# import scipy.io as sio
# import scipy.sparse as spr
# 
# s = spr.csc_matrix((45,1))
# 
# import markov_chain as mc
# #data = str.split(open("../files/sample_corpus").read())
# data = "He thought at first to rederive it by first principles; for he'd decided to rage against the dying of the light, and for him this meant not fearing to believe that he was the most seminal mind who had ever existed.  It was not something which he believed natively, but he postulated that in order to recreate the world he had not choice but to assume that he could, and that no one else had ever believed it so hard as he had.  He thought at first to work from first principles, and arrived only again and again at Cogito ergo sum.  But the more he thought about that even the less certain that seemed.  Was that not contingent on some arbitrary definition that such a thing as 'I' exists?  Cogito ergo sum proved the existence of only one thing, and that thing was not the concept of I.  Therefore: I think, therefore something is.  But even that broke into pieces: whence the concept therefore?  Could one even assume that there was such thing as being?  And so instead, he decided to work from second principles."#[1, 2, 1, 2, 3, 2, 3, 2, 3, 2, 1]
# model = mc.MarkovChain(data, 1)
# model.learn_model(data, generative=1)
# 
# s = model.generate_sequence(seed="", k = 1000)
# 
# print model.l_avg_l("Why is there pain?")
# print model.ll_of_data_instance("that's a pity, Are you ok?")
# print model.ll_of_data_instance("that's a pity,Are you ok?")
# print model.l_avg_l("that's a pity, are you ok.  maybe I'll say it in german")
# print model.l_avg_l("espanol me duele no poder escribir la tilda.  si solamente me gustara massted")
# print model.l_avg_l("cccccccccccccccccccccccc")
# 
# print s
#===============================================================================