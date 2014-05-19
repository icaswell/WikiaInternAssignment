'''
Created: 4 April 2014
@author: Isaac Caswell


'''

import numpy as np
import random as rand
import scipy.sparse as spr

class MarkovChain:
    """
    ARGUMENTS
    data: a string
    markov_level: the amount of tokens per value (BAD WORDING REDO) in the Markov model.  
                [explain more]
    
    TODO:    a) make it so data can be a list of strings (i.e. a document)
             b) make it so the alphabet can be given instead of having to compute it
             in the function?  but that would be a minimal speedup.
    
    ************
    CONVENTIONAL NAMES
    C: the number of characters in the alphabet.
    N: the number of values that each state in the markov chain can take on.
        pragmatically, this is size(alphabet)^TOTOTOTO_length
    ************  
    INTERNAL VARIABLES:  
    self._counts: a dict where _counts[X_i][j] is the MLE for P(c_j|X_i), or in other words,
        the counts in the data in which token c_j occurs after sequence of tokens X_i.
    self._transition: the transition matrix, here in dict form, where _transition[X_i][j]
        is the estimate for P(c_j|X_i).  It is simply self._counts, normalized (and maybe smoothed.)
    self._error_token: a character to indicate that something has gone wrong while 
        generating a sequence.  Essentially a debugging tool.  Cannot be the empty string, 
        or infinite loop may occur.
    """
    def __init__(self, data, markov_level, alphabet = False, token_smooth = 1.0, sequence_smooth = 1.0):
        """
        
        """
        if markov_level>len(data)+1: 
            print "ERROR: markov_level>len(data)+1."
            return
        
        if type(data) == type(''):
            self._data_is_string_type = True
        else: 
            self._data_is_string_type = False
            
        self._token_smooth = token_smooth
        self._sequence_smooth = sequence_smooth
        
        #TODO: this line no longer needed now that learn_model is not in constructor, yeah??
        #data = tuple(data) #if the input is a list, this is needed for the hashing.

        self.markov_level = markov_level
        
        #It would be more efficient to calculate this inside learn_model, but 
        #if the alphabet is given and is a strict superset of the tokens in the data, 
        #we'll run into problems. 
        self._alphabet = alphabet
        if not alphabet:
            self._alphabet = list(set(data))
        
        self._token_to_index = {}
        for i in range(len(self._alphabet)):
            self._token_to_index[self._alphabet[i]] = i
        
        self._error_token = "#"
        
        #self.learn_model(data, generative)
        
        

    #================================================================#
    #   LEARNING
    #================================================================#
    


    def learn_model(self, data, generative=0):
        """
        """
        data = tuple(data)#if the input is a list, this is needed for the hashing.
        
        counts = {}
        sequence_priors = {}
        token_priors = {}
        for token in self._alphabet:
            token_priors[token] = 0 
        
        for i in range(0, len(data)-self.markov_level):
            X_i = data[i:i+self.markov_level]
            next_token = data[i+self.markov_level]            
            
            if X_i not in counts:
                #consider a dict (no conversion to indices, but more annoying normalization)
                counts[X_i] = spr.csc_matrix((len(self._alphabet), 1)) 
                sequence_priors[X_i] = 0
                
            counts[X_i][self._token_to_index[next_token], 0] +=1
            sequence_priors[X_i] += 1
            token_priors[next_token] +=1
    
        if generative:
            """If the model will be used to generate random sequences.
            Note that sequence_priors is being split to seed_bucket_dist
            and seeds only for the purpose of sampling an initial seed for a generated sequence.
            (One cannot sample from the dict of priors.)
            This means that new sequences can only start from seeds in 
            the data, which seems OK to me.
            """
            seeds = []
            seed_bucket_dist = []
            for seed in sequence_priors:
                seeds.append(seed)
                seed_bucket_dist.append(sequence_priors[seed])
            self._seeds = seeds
            self._seed_bucket_dist = np.array(seed_bucket_dist)*1.0 / sum(seed_bucket_dist)
            
        if data[-self.markov_level:] not in counts:
            print "UNCERTAINTY IN DATA: if the sequence \'%s\' occurs," \
            "undefined behavior will ensue."%str(data[-self.markov_level:])
            counts[data[-self.markov_level:]] = spr.csc_matrix((1,1))#TODO: arbitrary magic!   
        
        self._transition = counts #NOTE THAT IT'S NOT normalized!
        
        sequence_cardinality = len(self._alphabet)**self.markov_level
        self._sequence_priors, self._unobserved_sequence_prior = \
                        self._normalize_dict(sequence_priors, sequence_cardinality, self._sequence_smooth)
        
        self._token_priors = self._normalize_dict(token_priors, len(self._alphabet), self._token_smooth)[0]

    
    def _normalize_dict(self, d, sequence_cardinality, smooth):
        """
        takes in a dict with numeric values, representing a probability distribution over the keys,
        and returns a tuple of (the dict with normalized keys, default prior value for a key not in the dict).
        Note that the dict is normalized over the space of all keys (which has cardinality sequence_cardinality), 
        although not all keys are present in the dict.
        """
        smooth = smooth*1.0 #just in case...
        total = 0 + sequence_cardinality * smooth
        for key in d:
            total +=d[key]
            
        for key in d:
            d[key] = (d[key] + smooth)/total
            
        return (d, smooth/total)
    
    
    #================================================================#
    #   LIKELIHOOD CALCULATION
    #================================================================#
      
          
    def ll_of_data_instance(self, data_instance):
        ll = 0.0
        
        # add the log of the prior probability of first sequence 
        initial_sequence = tuple(data_instance[0:self.markov_level])
        if initial_sequence in self._sequence_priors:
            ll = np.log(self._sequence_priors[initial_sequence])
        else:
            ll = np.log(self._unobserved_sequence_prior)
        
        for i in range(self.markov_level, len(data_instance)):
            key = tuple(data_instance[i-self.markov_level:i])
            if key not in self._transition:
                ll += np.log(self._token_priors[data_instance[i]])
            else:
                p_token_given_seq = self._prob_from_sparse_unnorm_bucket_dist(self._transition[key], \
                                                                       self._token_to_index[data_instance[i]], \
                                                                       self._token_smooth)
                ll += np.log(p_token_given_seq)
        return ll
        
        
    def l_avg_l(self, data_instance):
        """log of the Mth root of the likelihood, where M is the amount of tokens in the 
        data instance.  Useful if comparing two different data instances
        of differing length.
        """
        M = len(data_instance)
        return self.ll_of_data_instance(data_instance)/M
    
            
    def _prob_from_sparse_unnorm_bucket_dist(self, dist, idx, smooth):
        """
        dist is a sparse array.
        """
        cardinality = sum(dist.get_shape())
        
        return (dist[idx, 0] + smooth)/(dist.sum() + smooth*cardinality)
        
    
    
    #================================================================#
    #   SEQUENCE GENERATION
    #================================================================#
    
    
    def generate_sequence(self, seed="", k = 100):
        """
        if seed is the empty string, the MLE frequency distribution is sampled
        for the seed.
        k: the length of the generated sequence.
        """
        print "DEBUG: set random seed"
        rand.seed(1)
        cur = seed
        if seed == "":
            cur = self._choose_seed()
        generated_sequence = list(cur)
        for i in range(k):
            if (self._transition[cur].nonzero()[0]).size == 0:
                print "generation terminated because the current sequence is the last one " \
                "in the data, and I didn't know what else to do"
                break
            next = self._next_token(cur)
            generated_sequence.append(next)
            cur = tuple((list(cur) + [next])[1:])
            
        if self._data_is_string_type:
            return reduce(lambda x, y: x+y, generated_sequence)
        return generated_sequence
             
    def _sample_bucket_distribution(self, bucket_dist):
        """performs a simple sampling of an index from an array representing a discrete 
        probability distribution.  (i.e. a CPD over values a variable can take on)""" 
        r = rand.random()
        for i in range(len(bucket_dist)):
            if r<bucket_dist[i]:
                return i
            r -= bucket_dist[i]
        print "ERROR! NOT NORMALIZED!!"
        return -1
    
    def _sample_unnorm_bucket_dist_sparse(self, bucket_dist):
        """performs a simple sampling of an index from an array representing a discrete, unnormalized 
        probability distribution.  (i.e. a CPD over values a variable can take on)""" 
        normal_dist = bucket_dist/bucket_dist.sum()
        r = rand.random()
        for i in normal_dist.nonzero()[0]:
            if r<normal_dist[i, 0]:
                return i
            r -= normal_dist[i, 0]
        print "ERROR! NOT NORMALIZED!! WTF??"
        return -1
        
    def _next_token(self, sequence):
        bucket_dist = self._transition[sequence]
        token_idx = self._sample_unnorm_bucket_dist_sparse(bucket_dist)
        if token_idx <0:
            return self._error_token;
        return self._alphabet[token_idx]
    
    def _choose_seed(self):
        return self._seeds[self._sample_bucket_distribution(self._seed_bucket_dist)] 
    
    