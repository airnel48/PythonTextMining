# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:03:11 2017

@author: Eric Nelson
M07296609
"""

import nltk
#nltk.download()
from nltk.book import *

#13 What % of noun synsets have no hyponyms? Use wn.all_synsets('n')

from nltk.corpus import wordnet as wn
all_nouns=list(wn.all_synsets('n'))
have_hypo=[w for w in wn.all_synsets('n') if len(w.hyponyms())>0]
pct_w_no_hypo=(1-(len(set(have_hypo))/len(set(all_nouns))))
#ANSWER=79.67%

#14 define a fxn supergloss(s) that takes a synset s as its argument 
#and returns a string consisting of the concatenation of the definition
#of s and the definitions of all the hypernyms and hyponyms of s

def supergloss(s):
    dfntn=[wn.synset(s).definition()]
    hpernm=wn.synset(s).hypernyms()
    hpernmdef=[i.definition() for i in hpernm]
    hponm=wn.synset(s).hyponyms()
    hponmdef=[i.definition() for i in hponm]
    fnl=dfntn+hpernmdef+hponmdef
    print(fnl)
    
supergloss('car.n.01')

#16 write a program to generate table of lexical diversity scores
#(token/type ratios) as we saw in 1.1. Include full set of Brown Corpus
#genres (nltk.corpus.brown.categories()). Which genre has the lowest
#diversity (greatest number of tokens per type)? Is that what you would
#have expected?

#should have genre, tokens, types, and lexical diversity = types/tokens
    
from nltk.corpus import brown
for genre in brown.categories():
    tokens=len(brown.words(categories=genre))
    types=len(set(brown.words(categories=genre)))
    diversity=types/tokens
    print (genre,tokens,types,diversity)
#ANSWER = learned has the lowest diversity. I might have expected a simpler text like romance to have lower diversity
#but romance is not far behind.

#17 write a fxn that finds 50 most freq words of a text that are
# not stopwords

def wrd_freq(text):
    stopwords=nltk.corpus.stopwords.words('english')
    words=[w.lower() for w in text if w not in stopwords and w.isalpha()]
    fd=nltk.FreqDist(words)
    return fd.most_common(50)

wrd_freq(text5)

#18 write program to print 50 most freq bigrams of a text, omitting
#bigrams that contain stopwords

def bi_freq(text):
    stopwords=nltk.corpus.stopwords.words('english')
    words=[w.lower() for w in text if w not in stopwords and w.isalpha()]
    bigrams=nltk.bigrams([w.lower() for w in words])
    fd=nltk.FreqDist(bigrams)
    return fd.most_common(50)

bi_freq(text4)
    
#20 write a fxn word_freq() that takes a word and the name of a section
#of the Brown Corpus as arguments and computes freq of the word in that
#section

def word_freq(word,section):
    text=brown.words(categories='religion')
    textlower=[w.lower() for w in text]
    fdist=nltk.FreqDist(textlower)
    wordfreq=fdist[word]
    return print("the word '",word,"' occurs ",wordfreq," times in the '",section,"' section of Brown Corpus")
    
word_freq('the','religion')

#22 define a fxn hedge(text) which processes a text and produces a new
#version with the word 'like' between every third word
        
def hedge(text):
    lst=list(text)
    n=-1
    for i in lst:
        print(i)
        n=n+1
        if n % 3 == 2:
            print('like')

hedge(sent2)


