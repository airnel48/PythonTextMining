# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:30:52 2017

@author: Eric
"""

#13 Write code to initialize a two-dimensional array of sets called word_vowels and process a list of words, 
# adding each word to word_vowels[l][v] where l is the length of the word and v is the number of vowels it contains.

def wrdvwls(sentence):
    m=len(sentence)
    n=2
    word_vowels = [[set() for i in range(n)] for j in range(m)]
    w=0  
    for a in sentence:
        v=0
        l=len(a)
        word_vowels[w][0].add(l)
        for b in a:
            if b.lower() in "aeiou":
                v=v+1
        word_vowels[w][1].add(v)
        w=w+1
    print(word_vowels)
   
sentence = ['Today', 'is', 'April', '5th']
wrdvwls(sentence)    
        
#15 Write a program that takes a sentence expressed as a single string, splits it and counts up the words.
# Get it to print out each word and the word's frequency, one per line, in alphabetical order.

from nltk import *

def wordfreq(sentence):
	words = sentence.split(' ')
	count = len(words)
	freq = FreqDist(words)
	freq2 = sorted(freq.most_common(count))
	for a in freq2:
		print(a)

sentence = 'Today is April 5th 2017 and I turn 28 on April 8th 2017'
wordfreq(sentence)

#19 Write a list comprehension that sorts a list of WordNet synsets for proximity to a given synset. 
# For example, given the synsets minke_whale.n.01, orca.n.01, novel.n.01, and tortoise.n.01, 
# sort them according to their shortest_path_distance() from  right_whale.n.01.

from nltk.corpus import wordnet as w

def short_proximity(lst,synset):
    return [b for (a,b) in (sorted([(synset.shortest_path_distance(a), a) for a in lst]))]

short_proximity(whale, wn.synset('right_whale.n.01'))


#20 Write a function that takes a list of words (containing duplicates) and returns a list of words (with no duplicates) 
# sorted by decreasing frequency. E.g. if the input list contained 10 instances of the word table and 9 instances of the word chair, 
# then table would appear before chair in the output list.

from nltk.book import FreqDist

def wordsort(sentence):
    distcount = len(set(sentence))
    freq=FreqDist(sentence)
    freq2=list(freq.most_common(distcount))
    a = [b[0] for b in freq2]
    print(a)

sentence = ['hoosier', 'heaven', 'hockey', 'halloween', 'hoosier', 'hoosier', 'hoosier', 'heaven', 'halloween', 'heaven']
wordsort(sentence)

#21 Write a function that takes a text and a vocabulary as its arguments and returns the set of words that appear in the text but 
# not in the vocabulary. Both arguments can be represented as lists of strings. Can you do this in a single line, using set.difference()?

def textnotvocab(text,vocab):
    print(set(text).difference(vocab))
    
text=['How', 'much', 'wood', 'could', 'a', 'woodchuck', 'chuck', 'if', 'a', 'woodchuck', 'could', 'chuck', 'wood']
vocab=['wood', 'chuck']

textnotvocab(text,vocab)

#now in a single line
set(['How', 'much', 'wood', 'could', 'a', 'woodchuck', 'chuck', 'if', 'a', 'woodchuck', 'could', 'chuck', 'wood']).difference(['wood', 'chuck'])

#22 Import the itemgetter() function from the operator module in Python's standard library (i.e. from operator import itemgetter). 
# Create a list words containing several words. Now try calling: sorted(words, key=itemgetter(1)), and  sorted(words, key=itemgetter(-1)). 
# Explain what itemgetter() is doing.

from operator import itemgetter

words = ['zebra','zebra','ant','bat','cat','dog','ant','bat','cat','dog','ant','bat','ant']

words=['Eric','Arthur','Nelsons','Eric','Eric','Arthur']

words=['four','one', 'three', 'two', 'two', 'three', 'four', 'four','three','four' ]

sorted(words, key=itemgetter(1))
sorted(words, key=itemgetter(-1))

#In this example, itemgetter is determing which letter in the words to sort them by.
#Traditionally, sorted would sort by the first, then second, then third letter and so on. 
#When we set itemgetter to 1 it obtains the second letter of each word and sorts them alphabetically. 
#Ties are settled by the original order of the words, not the next letter in the word. 
#When we ste itemgetter to -1 it obtains the last letter of each word and sorts them alphabetically.
#Again, ties are settled by the original order of the words, not the next letter in the word.

#23 Write a recursive function lookup(trie, key) that looks up a key in a trie, and returns the value it finds. 
# Extend the function to return a word when it is uniquely determined by its prefix (e.g. vanguard is the only word that starts with vang-, 
# so lookup(trie, 'vang') should return the same thing as lookup(trie, 'vanguard')).

#trie creation from class notes                                                                                     
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value
trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
insert(trie, 'vanguard', 'stocks')
trie=dict(trie)                                                                                     
            
#custom function                                                                         
def lookup(trie, key):
    trie2 = trie
    for a in key:
        trie2 = trie2[a]
    return trie2
    
lookup(trie,'cha')                                                                                        
                                                                                     