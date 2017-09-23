# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:41:17 2017

@author: Eric Nelson M07296609
"""

#2 Using any of the three classifiers described in this chapter, and any features you can think of, build the best 
#name gender classifier you can. Begin by splitting the Names Corpus into three subsets: 500 words for the test set, 
#500 words for the dev-test set, and the remaining 6900 words for the training set. Then, starting with the example 
#name gender classifier, make incremental improvements. Use the dev-test set to check your progress. Once you are 
#satisfied with your classifier, check its final performance on the test set. How does the performance on the test set 
#compare to the performance on the dev-test set? Is this what you'd expect?

import nltk
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["first_letters"] = name[:2].lower()
    features["length"] = len(name)
    features["last_letter"] = name[-1].lower()
    features["last_letters"] = name[-3:].lower()
    for letter in 'aeiouy':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
    
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]

dev_test_set, test_set, train_set = featuresets[:500], featuresets[500:1000], featuresets[1000:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.show_most_informative_features(50)

print(nltk.classify.accuracy(classifier, dev_test_set))
#after much comparison between train and dev_test and reworking the model, the dev_test results are 0.824
print(nltk.classify.accuracy(classifier, test_set))
#the results from test_set which was never used when building the model is 0.812
#it would make sense that our test_set performance would be slightly worse than dev_test_set
#We built the model on train and test on dev_test_set and reworked the model until we had good
#results on both train and dev_test_set. However, that means we probably still overfit a bit to those two
#datasets. We would expect to perform slightly worse on the test_set and that is what happened.


#3 The Senseval 2 Corpus contains data intended to train word-sense disambiguation classifiers. It contains data for 
#four words: hard, interest, line, and serve. Choose one of these four words, and load the corresponding data:

#Using this dataset, build a classifier that predicts the correct sense tag for a given instance. See the corpus HOWTO 
#at http://nltk.org/howto for information on using the instance objects returned by the Senseval 2 Corpus.

import nltk
from nltk.corpus import senseval
import random
#I chose to use the word 'serve'
instances = senseval.instances('serve.pos')
size = int(len(instances) * 0.1)

for inst in instances[:5]:
    p = inst.position
    left = ' '.join(w for (w,t) in inst.context[p-2:p])
    word = ' '.join(w for (w,t) in inst.context[p:p+1])
    right = ' '.join(w for (w,t) in inst.context[p+1:p+3])
    senses = ' '.join(inst.senses)
    
def features(instance):
    feat = dict()
    p = instance.position
       ## previous word and tag
    if p: ## > 0
        feat['wp'] = instance.context[p-1][0]
        feat['tp'] = instance.context[p-1][1]
       ## use BOS if it is the first word
    else: # 
        feat['wp'] = (p, 'BOS')
        feat['tp'] = (p, 'BOS')
       ## following word and tag       
        feat['wf'] = instance.context[p+1][0]
        feat['tf'] = instance.context[p+1][1]
    return feat


featureset =[(features(i), i.senses[0]) for i in 
             instances if len(i.senses)==1]

### shuffle them randomly
random.shuffle(featureset)
train, test = featureset[size:], featureset[:size]
classifier = nltk.NaiveBayesClassifier.train(train)
print (nltk.classify.accuracy(classifier, train))
#0.710
print (nltk.classify.accuracy(classifier, test))
#0.661


#4 Using the movie review document classifier discussed in this chapter, generate a list of the 30 features that the 
#classifier finds to be most informative. Can you explain why these particular features are informative? Do you find 
#any of them surprising?
import random
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
    
classifier.show_most_informative_features(30)
#of the top 30 feature results from the classifier, some words are easy to identify why they are informative
#from a negative review standpoint, it makes sense that terms like amateurish, nagging, and fluke would be 
#strong indicators of a negative review. On the other side, terms like palpable, layered, and indelible make
#sense for reviews that are positive about a movie. 
#Some surprising results include that unfairly (positive), weaknesses (positive), and dread (positive) are
#categorized as they are. It might help if we knew the context of these words or at least the words that came
#before and aftre each of them. For example, if the word weaknesses is preceeded by the word 'few' than a negative
#word takes on a positive connotation. 


#5 Select one of the classification tasks described in this chapter, such as name gender detection, document 
#classification, part-of-speech tagging, or dialog act classification. Using the same training and test data, and the 
#same feature extractor, build three classifiers for the task: a decision tree, a naive Bayes classifier, and a Maximum 
#Entropy classifier. Compare the performance of the three classifiers on your selected task. How do you think that your 
#results might be different if you used a different feature extractor?

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["first_letters"] = name[:2].lower()
    features["length"] = len(name)
    features["last_letter"] = name[-1].lower()
    features["last_letters"] = name[-3:].lower()
    for letter in 'aeiouy':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
    
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
test_set, train_set = featuresets[:500], featuresets[500:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, train_set))
#0.84
print(nltk.classify.accuracy(classifier, test_set))
#0.82

classifier2 = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier2, train_set))
#.96
print(nltk.classify.accuracy(classifier2, test_set))
#.744

classifier3 = nltk.MaxentClassifier.train(train_set)
print(nltk.classify.accuracy(classifier3, train_set))
#0.88
print(nltk.classify.accuracy(classifier3, test_set))
#0.812

# In this scenario, the Naive Bayes Classifier reigned supreme. This was while using the first letter, first two
#letters, lenth, last letter, last two letters, and count of vowels to classify a given name's gender. The shear 
#number of possible features causes the decision tree to strongly overfit on the training data and perform much
#worse on the test data. If we limited the number of features therefor limiting the branches in the tree, it would
#likely have less of an overfitting issue. I would have expected the Naive Bayes to perform worse out of sample given
#that the model assumes the features are independent and in this case there are many features that are highly dependent
#such as the first letter and first 3 letters of a name and the last letter and last 3 letters of a name. 
#There is a chance the naive bayes would perform better if we kept the features more independent. 
#In this caes, even though the entropy classifier is similar to the naive bayes and runs iteratively, it had more
#significant overfitting issues. 


#6 The synonyms strong and powerful pattern differently (try combining them with chip and sales). What features are 
#relevant in this distinction? Build a classifier that predicts when each word should be used.

from nltk.util import ngrams
from nltk.corpus import brown
import random
words = brown.words()
bigs = list(nltk.bigrams(words))
trigrams=list(ngrams(words,3))

power_gram = [(a, c) for (a, b, c) in trigrams if b in ('powerful')]
strong_gram = [(a, c) for (a, b, c) in trigrams if b in ('strong')]

labels = ([(a, b, 'powerful') for (a,b) in power_gram] + [(a, b, 'strong') for (a,b) in strong_gram])
random.shuffle(labels)

def features(before, after):
    features={}
    features["pre"]=before
    features["post"]=after
    return features

sets=[(features(a,b), word) for (a, b, word) in labels]

dev_test, test, train = sets[:500], sets[500:1000], sets[1000:]

classifier = nltk.NaiveBayesClassifier.train(train)
classifier.show_most_informative_features(50)

print(nltk.classify.accuracy(classifier, train))
#0.984
print(nltk.classify.accuracy(classifier, dev_test))
#0.942 accuracy
print(nltk.classify.accuracy(classifier, test))
#0.916 accuracy

#Looking at the words that come before and after the words 'powerful' and 'strong', we can predict which
#word should be used in the context with over 90% accuracy. Top indicators include preceeding the 
#word 'the' (365x strong), preceding the word 'have' (260x powerful), following the word 'which' (86x powerful), 
#and following the word 'if' (71x powerful). 








import nltk
from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["first_letters"] = name[:2].lower()
    features["length"] = len(name)
    features["last_letter"] = name[-1].lower()
    features["last_letters"] = name[-3:].lower()
    for letter in 'aeiouy':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
    
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]

dev_test_set, test_set, train_set = labels[:500], labels[500:1000], labels[1000:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(50)

print(nltk.classify.accuracy(classifier, dev_test_set))
print(nltk.classify.accuracy(classifier, test_set))




import nltk
from nltk.corpus import brown
words = brown.words()
size = int(len(words) * 0.1)
train, test = words[size:], words[:size]

 def pos_features(sentence, i, history): [1]
     features = {"suffix(1)": sentence[i][-1:],
                 "suffix(2)": sentence[i][-2:],
                 "suffix(3)": sentence[i][-3:]}
     if i == 0:
         features["prev-word"] = "<START>"
         features["prev-tag"] = "<START>"
     else:
         features["prev-word"] = sentence[i-1]
         features["prev-tag"] = history[i-1]
     return features

class ConsecutivePosTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence,  history)

	
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
print(tagger.evaluate(test_sents))        
        
pos_features()
tagger=ConsecutivePosTagger(train)
print(tagger.evaluate(test))






















import random

import nltk
from nltk.corpus import brown
words = brown.words()
random.shuffle(words)
size = int(len(words) * 0.1)
train, test = words[size:], words[:size]


all_words = nltk.FreqDist(w.lower() for w in words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(document_features(brown.words('powerful')))

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

