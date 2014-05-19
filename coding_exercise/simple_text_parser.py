"""
@Author: Isaac Caswell
5 May 2014

Provides a method to tokenize a doc.  Upon further reflection there is probably 
a module to do this much more intelligently somewhere, but this is free and easy enough, 
and something which I often want done so I might as well make meself a nice tool for it!
"""


import string

#=============================#
#PUBLIC METHODS
#=============================#


def tokenize_string_to_list(doc, stopwords_file = '../files/stopwords', remove_caps = 1, remove_punc = 1, remove_stopwords = 1):
    result = []
    """
    remove_caps == 1:  tokens like S.H.I.E.L.D. ; X-Force  become shield, xforce
    remove_punc == 2: preserves these tokens.
    """
    if remove_caps:
        doc = doc.lower()
    punc = []
    if remove_punc:
        punc = string.punctuation
        
    doc_list = list(doc)
    if remove_punc == 2:
        l = len(doc_list)
        doc_list.append(' ')
        doc_list =  ''.join([doc_list[i] for i in range(l) if not doc_list[i] in punc or doc_list[i+1] != ' ']).split()
    elif remove_punc == 1:
        doc_list =  ''.join([ch for ch in doc_list if not ch in punc]).split()
    else:
        doc_list = "rabbit" #this had better not happen
        
    #doc_list re.findall('\w+', doc)

    if remove_stopwords:
        doc_list = _remove_stopwords_from_doc(doc_list, stopwords_file)
    
    return doc_list


def tokenize_string_to_file(doc, f_dest, stopwords_file = '../files/stopwords', remove_caps = 1, remove_punc = 1, remove_stopwords = 1):
    doc_list = tokenize_string_to_list(doc, stopwords_file, remove_caps, remove_punc, remove_stopwords)
    doc = " ".join(doc_list)
    open(f_dest, 'w').write(doc)
    

def tokenize_file_to_list(f_src, stopwords_file = '../files/stopwords', remove_caps = 1, remove_punc = 1, remove_stopwords = 1):
    doc = open(f_src, 'r').read()
    return tokenize_string_to_list(doc, stopwords_file, remove_caps, remove_punc, remove_stopwords)


def tokenize_file_to_file(f_src, f_dest, stopwords_file = '../files/stopwords', remove_caps = 1, remove_punc = 1, remove_stopwords = 1):
    doc = open(f_src, 'r').read()
    doc_list = tokenize_string_to_list(doc, stopwords_file, remove_caps, remove_punc, remove_stopwords)
    doc = " ".join(doc_list)
    open(f_dest, 'w').write(doc)


#=============================#
#PRIVATE METHODS
#=============================#
    
def _remove_stopwords_from_doc(doc, stopwords_file):
    stop_words = [];
    with open(stopwords_file) as f:
        stop_words = f.readlines()
    stop_words = set(map(str.strip, stop_words))
    return [word for word in doc if not word in stop_words]


#=============================#
#TESTS
#=============================#

#tokenize_file_to_file("../files/temp", "../files/temp2")

