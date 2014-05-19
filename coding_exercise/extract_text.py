#!/usr/bin/python
"""
 
#technical words from http://meta.wikimedia.org/wiki/Stop_word_list

#stop words from http://meta.wikimedia.org/wiki/MySQL_4.0.20_stop_word_list

The main public function this module exports is parse_and_save_corpus_to_file(..), which runs the top 
articles for the universe in question through a simpler tokenizer, and writes them to a file
(format: one per line, tokens separated by spaces).  Note that documents of length less than 
MIN_DOC_LENGTH are excluded.  (such articles might for example be mostly or entirely images)

 """
 
import requests
import os
import simple_text_parser as stp

##This is a pretty arbitrary number, but I include it to cull for 'articles' such as 'superman titles' or 
##the empty string, resulting from an 'article' that was for instance actually an image
MIN_DOC_LENGTH = 100


#===============================================================================
# PUBLIC METHODS:
#===============================================================================

def parse_and_save_corpus_to_file(f_dest, universe, hold_out_pct = 0, debug = {"percent_articles_to_consider":1}):
    """
    reads all the top articles from the given universe (e.g. 'dc' or 'marvel'), tokenizes them and 
    reads them into the file specified by f_dest.  Uses generators to be easy on RAM.
    """
    corpus = corpus_generator(universe, debug)
    corpus = (stp.tokenize_string_to_list(doc) for doc in corpus)
    
    lines_in_corpus = 0
    clear_file(f_dest)
    with open(f_dest, 'a') as f:
        for doc in corpus:
            lines_in_corpus +=1
            doc_as_string = " ".join(doc)
            if len(doc_as_string) >= MIN_DOC_LENGTH:
                doc_as_string = doc_as_string.encode('utf-8')
                f.write(doc_as_string)
                f.write('\n')
        prune_trailing_endline(f)
    
    if hold_out_pct !=0:
        """create a training and a testing corpus"""
        f_dest_test = f_dest + '_test'
        f_dest_train = f_dest + '_train'
        clear_file(f_dest_test)
        clear_file(f_dest_train)
          
        with open(f_dest, 'r') as f_total:  
            with open(f_dest_test, 'a') as f_test:
                transfer_n_lines(f_total, f_test, int(lines_in_corpus*hold_out_pct))
            with open(f_dest_train, 'a') as f_train:
                transfer_n_lines(f_total, f_train, lines_in_corpus - int(lines_in_corpus*hold_out_pct))
           
 
#===============================================================================
# PRIVATE
#===============================================================================

def get_paragraphs_from_json(content):
    """
    this function assumes that all Wikia articles have the same form, 
    namely only two levels of nesting.
    """
    paragraphs = []
    for section in content['sections']:
        for elem in section['content']:
            if elem[u'type'] == u'paragraph':
                paragraphs.append(elem)
    return paragraphs


def agglomerate_doc(doc):
    """
    takes a document (Wikia article) represented as a list of paragraph objects, and 
    returns a string which is the concatenation of the text associated with teach paragraphs
    """
    string_paragraphs = [paragraph['text'] for paragraph in doc]
    if not string_paragraphs:
        return ""
    return reduce(lambda x, y: x+y, string_paragraphs)

class corpus_generator():
    """
    A generator for retrieving articles from Wikia for a particular universe.
    
    debug:
    The defaults for debug will lead to normal (nondebugged) performance.  Setting
    debugging variables is useful for testing:
        percent_articles_to_consider - the percent of articles the generator will iterate over.
                    for instance, if there are 244 top articles, and this value is .01, this 
                    function will operate on 2 articles.
    """
    
    def __init__(self, universe, debug = {"percent_articles_to_consider":1}):        
        self._simpleJson_request_url = "http://%s.wikia.com/api/v1/Articles/AsSimpleJson"%universe
        req = requests.get("http://%s.wikia.com/api/v1/Articles/Top"%universe)
        self._ids_of_top_articles = [i['id'] for i in req.json()['items']]
        articles_to_consider = len(self._ids_of_top_articles)
        self._articles_to_consider = int(debug["percent_articles_to_consider"] * articles_to_consider)
        
    def __iter__(self):
        for i in range(self._articles_to_consider):
            req = requests.get(self._simpleJson_request_url, params = {'id':self._ids_of_top_articles[i]})
            doc = get_paragraphs_from_json(req.json())
            doc = agglomerate_doc(doc)
            yield doc

def prune_trailing_endline(f):
    f.seek(-1, os.SEEK_END)
    f.truncate()

def clear_file(fname):
    with open(fname, 'w') as f:
        f.write('') #clear the file just in case       
        
def transfer_n_lines(f_src, f_dest, n): 
    """writes the first n lines from f_src into f_dest"""
    for i in range(n):
        line = f_src.readline()
        f_dest.write(line)
    prune_trailing_endline(f_dest)

#===============================================================================
# TESTS:
#===============================================================================
          
#parse_and_save_corpus_to_file("../files/marvel_corpus", "marvel", hold_out_pct = .3, debug = {"percent_articles_to_consider":.1})





#===============================================================================
# APPENDIX:
#===============================================================================


#==================
#1) get_corpus as a list instead of a generator:
#=================
# 
# def get_corpus(universe, debug = {"percent_articles_to_consider":1}):
#     """description of parameters
#     """
#     simpleJson_request_url = "http://%s.wikia.com/api/v1/Articles/AsSimpleJson"%universe
#     req = requests.get("http://%s.wikia.com/api/v1/Articles/Top"%universe)
#     ids_of_top_articles = [i['id'] for i in req.json()['items']]
#     
#     corpus = []
#     articles_to_consider = len(ids_of_top_articles)
#     articles_to_consider = int(debug["percent_articles_to_consider"] * articles_to_consider)
#     print articles_to_consider
#     for i in range(articles_to_consider):
#         req = requests.get(simpleJson_request_url, params = {'id':ids_of_top_articles[i]})
#         doc = get_paragraphs_from_json(req.json())
#         doc = agglomerate_doc(doc)
#         corpus.append(doc)
#     return corpus
#===============================================================================


