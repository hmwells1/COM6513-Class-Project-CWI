import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
#trigrams
from nltk.util import ngrams
import collections
#synonyms
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
#strip accents
import unidecode
# syllables
import re
import pyphen
dic = pyphen.Pyphen(lang='en')
#unigrams
from collections import Counter
#POS
import spacy
nlpE = spacy.load('en')
nlpS = spacy.load('es')
#cross val
# from sklearn.model_selection import RandomizedSearchCV

class MyLine(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
            
        self.model = RandomForestClassifier(n_estimators = 100, min_samples_split = 5,
                                             min_samples_leaf= 1, max_features = 'auto', max_depth = 110,
                                             bootstrap = False,random_state = 5) 
        
    def extract_features(self, d):
        # add new features here to go in dictionary
        features_dict = {}
        
        w = d['target_word']
        s = d['sentence']
        
        word = w
        sent = s
        
        ######### Features for English
        if self.language == 'english':
        
            # given features
            features_dict['len_chars'] = len(word) / self.avg_word_length
            features_dict['len_tokens'] = len(word.split(' '))        
            
            # syllables
            features_dict['syllables'] = len(re.findall(r'[\w]+',dic.inserted(word)))
            
            # rare letter count
            features_dict['rare7_letters_sum'] = sum(map((word.lower()).count, "qjzxvkw"))
            
            # trigram counts
            for i in self.trigrams(word):
                features_dict['tri_' + i] = self.tris[i]
                
            # bigram counts
            for i in self.bigrams(word):
                features_dict['bi_'+i] = self.bis[i]
                
            # number of synonyms
            features_dict['synonyms'] = self.synonyms(word)
            
            # hyponymns
            features_dict['hyponyms'] = self.hyponyms(word)
                
            word1 = nlpE(word)
            # Shape
            for i in range(len(word1)):
                features_dict['Shape' + word1[i].shape_] = self.S[word1[i].shape_]/self.total
        
            # dep
            for i in range(len(word1)):
                features_dict['DEP' + word1[i].dep_] = self.S[word1[i].dep_]/self.total
                
        ######### Features for Spanish
        if self.language == 'spanish':
            
            # given features
            features_dict['len_chars'] = len(word) / self.avg_word_length
            
            # trigram counts
            for i in self.trigrams(word):
                features_dict['tri_' + i] = self.tris[i]
            
            # unigrams
            features_dict['unigram'] = self.words[word]
        
             
        ######### All Attempted Features
        
        # given features
        # features_dict['len_chars'] = len(word) / self.avg_word_length
        # features_dict['len_tokens'] = len(word.split(' '))
        
        # syllables
        # features_dict['syllables'] = len(re.findall(r'[\w]+',dic.inserted(word)))
        
        # letter counts
        # features_dict['vowel_sum'] = sum(map((word.lower()).count, "aeiou"))
        # features_dict['consonant_sum'] = sum(map((word.lower()).count, "qwrtypsdfghjklzxcvbnm"))
        # features_dict['rare7_letters_sum'] = sum(map((word.lower()).count, "qjzxvkw"))
        
        # length modulo 8
        # features_dict['len_modulo_8'] = len(word)%8
        
        # trigram counts
        # for i in self.trigrams(word):
            # features_dict['tri_' + i] = self.tris[i]
        
        # bigram counts
        # for i in self.bigrams(word):
            # features_dict['bi_'+i] = self.bis[i]
            
        # number of synonyms
        # features_dict['synonyms'] = self.synonyms(word)
        
        # hyponymns
        # features_dict['hyponyms'] = self.hyponyms(word)
        
        # unigrams
        # features_dict['unigram'] = self.words[word]
        
        # POS
        # if self.language =='english':
            # word1 = nlpE(word)
        # elif self.language == 'spanish':
            # word1 = nlpS(word)
        # for i in range(len(word1)):
            # features_dict['POS' + word1[i].pos_] = self.S[word1[i].pos_]/self.total
    
        # Shape
        # for i in range(len(word1)):
            # features_dict['Shape' + word1[i].shape_] = self.S[word1[i].shape_]/self.total
    
        # dep
        # for i in range(len(word1)):
            # features_dict['DEP' + word1[i].dep_] = self.S[word1[i].dep_]/self.total

        # text
        # for i in range(len(word1)):
            # features_dict['TEXT' + word1[i].text] = self.S[word1[i].text]/self.total
            
        # vector_norm
        # for i in range(len(word1)):
            # features_dict['VEC' + str(word1[i])] = self.S[word1[i]]

        # lemma
        # for i in range(len(word1)):
            # features_dict['LEMMA' + str(word1[i])] = self.S[word1[i].lemma_] 

        # prefix
        # for i in range(len(word1)):
            # features_dict['PREFIX' + str(word1[i])] = self.S[word1[i].prefix_] 

        # suffix
        # for i in range(len(word1)):
            # features_dict['SUFFIX' + str(word1[i])] = self.S[word1[i].suffix_] 


        return features_dict #[len_chars, len_tokens]
   
    # feature functions
    def unigram(self,text):
        d = Counter(text.split(' '))
        return d

    def trigrams(self,text):
        triList = []
        for i in text.lower().split(' '):
            chars = [c for c in i]
            triList.append((list(ngrams(chars,3))))
        triCount = []
        for i in triList:
            for j in i:
                triCount.append(''.join(j))
        return triCount
    
    def bigrams(self,text):
        biList = []
        for i in text.lower().split(' '):
            chars = [c for c in i]
            biList.append((list(ngrams(chars,2))))
        biCount = []
        for i in biList:
            for j in i:
                biCount.append(''.join(j))
        return biCount
    
    def synonyms(self, word):
        phrase = word
        syns = []
        for i in phrase.split(" "):
            syns.append(wn.synsets(i))
        size = 0
        for i in syns:
            size += len(i)
        return size

    def hyponyms(self,word):
        w = word.split(" ")
        s = 0
        for i in w:
            if len(wn.synsets(i)) != 0:
                n = wn.synsets(i)[0]
                s += len(n.hyponyms())
        return s

    def letter_freq(self,word):
        return Counter(list(word))
    
    # training function
    def train(self, trainset):
        self.words = Counter()
        self.bis = Counter()
        self.tris = Counter()
        self.S = Counter()
        self.total = 0
        self.letter_counter = Counter()
        self.total_features_dict = []
        self.vectorizer = DictVectorizer(sparse=False)
        for i in range(len(trainset)):
            
            if self.language == 'spanish':
                for w in trainset[i]['sentence'].split():
                    self.words[w] += 1
                
             
            for w in self.trigrams(trainset[i]['sentence']):
                self.tris[w] += 1
                

            if self.language == 'english':
                nlp = nlpE
                for w in self.bigrams(trainset[i]['sentence']):
                    self.bis[w] += 1
            # elif self.language == 'spanish':
                # nlp = nlpS
                sent = nlp(trainset[i]['sentence'])
                for token in sent:
                    self.S[token.shape_] +=1
                    self.S[token.dep_] +=1
                    self.total +=1
                    # self.S[token.pos_] +=1
                    # self.S[token.text] +=1
                    # self.S[token] = token.vector_norm
                    # self.S[token.lemma_] +=1
                    # self.S[token.prefix_] +=1
                    # self.S[token.suffix_] +=1
                
            self.total_features_dict.append(self.extract_features(trainset[i]))
                
        self.vec_maker = self.vectorizer.fit_transform(self.total_features_dict)
        
        
        
        X = np.zeros((len(trainset),(self.vec_maker).shape[1]))
        y = []
        
        for i in range(len(trainset)):
            X[i,:] = self.vectorizer.transform(self.extract_features(trainset[i]))
            y.append(trainset[i]['gold_label'])
    
        self.model.fit(X, y)
        
    # Testing function
    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.vectorizer.transform(self.extract_features(sent))[0])

        return self.model.predict(X)