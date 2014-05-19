"""
@author: Isaac Caswell
20 April 2014

use a Markov chain model to classify text

"""

import markov_chain as mc

class MarkovChainClassifier:
    def __init__(self, markov_level):
        self.markov_level = markove_level
    
    def learn_model(self, data):
        """
        data- a list of length N, where N is the amount of classes there are.  
              data[i] is a list of words representing an article or 
              agglomeration of articles.
              
        The class labels are implicitly 0, 1, 2, 3 .....
        """
        N = len(data)
        for i in range(N):
            model = mc.MarkovChain(data[i], 1)
            model.learn_model(data[i])
            self.models.append(model)
            
    
    
    def predictions(self, X_test):
        y_pred = []
        all_lls = [[m.ll_of_data_instance(X_i) for m in self.models] for X_i in X_test]
        return [lls.index(max(lls)) for lls in all_lls]
    
    def accuracy(self, X_test, y):
        """
        X_test- a list of data instances
        y- the class labels for these data instances
        """
        y_pred = self.predictions(X_test)
        return sum(y_test==y_pred)/len(y_test)*1.0 
            
            
            