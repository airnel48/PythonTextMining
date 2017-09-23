# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:15:18 2017

@author: Eric
"""

#18 read in some text from a corpus, tokenize it, and
#print the list of all wh-word types that occur (which, what, etc)
#print them in order. Are any words duplicated in thsi list
#because of the presence of case distinctions or punctuation?

from nltk.corpus import (gutenberg)
from nltk.tokenize import word_tokenize
import nltk
import re

tokens=word_tokenize(gutenberg.raw('blake-poems.txt'))
words = set([w for w in tokens])
print([w for w in words if re.search('^wh.', w)])

#19 create a file consisting of words and (made up) frequencies
#where each line consists of a word, the space character, and a positive
#integer (e.g. fuzzy 53). REad the file into a python list using 
#open(filename).readlines(). Next, break each line into its two fields
#using split(), and convert the number into an integer using int(). The 
#result should be a list of form [['fuzzy',53],...]

file = open('C:/Users/Eric/Desktop/IU.txt').readlines()
file_strip=[a.strip() for a in file]
file_split=[b.split(' ') for b in file_strip]
for c in file_split:
    c[1]=int(c[1])


#21 write a fxn unknown() that takes a URL as its argument and
#returns a list of unknown words that occur on that webpage. Extract all 
#substrings consisting of lowercase letters (using re.findall()) and 
#remove any items from this set that occur in the Words Coprus (nltk.corpus.words)
#try to categorize these words manually and discuss your findings

from nltk.corpus import words
from bs4 import BeautifulSoup
from urllib import request

def unknown(url):
    html=request.urlopen(url).read().decode('utf8')
    raw=BeautifulSoup(html).get_text()
    a=set(words.words())
    b=re.findall(r'\b[a-z]+',raw)
    c=[w for w in b if w not in a]
    d=sorted(set(c))
    print(d)
  
unknown('http://kenpom.com/blog/2018-pre-season-top-ten/')

#The site I have chosen is a blog for college basketball statistics. Overall, a majority of the unknown words are related to
#either basketball or colleges. Basketball terms include rebounding, steals, seasons, player names, etc. 
#College terms include the names of many universities with prominent basketball programs. 

#23 are you able to write a regular expression to tokenize text in 
#a way that the word "don't" is tokenized into do and n't? Explain why
#this regular expression won't work: <<n't|\w+>>

# <<n't|\w+>> won't work because it gets stuck on the single quote in n't 

html=request.urlopen('http://kenpom.com/blog/2018-pre-season-top-ten/').read().decode('utf8')
a=BeautifulSoup(html).get_text()
error=re.findall(r'\n't|\w+,a)
#we can get this to work if we escape on the single quote
b=re.findall(r'\b(do)(n\'t)', a)


#24 Try to write code to convert text into hAck3r using reg expr and
#substitution where e->3, i->1, o->0, l->|, s->5, .->5w33t!, ate=>.
#normalize txt to lowercase before converting. Add more substitutions
#of your own. no map so to two diff values: $ for word-initial s, and
#5 for word-internal s

text=('I will attend the University of Cincinnati in 8 years.')

def cnvrt(text):
	new_text = []
	for w in text:
		if re.findall(r'[eioslr8.]', w):
			if w == 'e':
				w = '3'
			elif w == 'i':
				w = '1'
			elif w == 'o':
				w = '0'
			elif w == 's':
				w = '5'
			elif w == 'l':
				w = '|'
			elif w == '.':
				w = '5w33t!'
			elif w == '8':
				w = 'ate'
			elif w == 'r':
				w = 'arrggg'
		new_text.extend(w)
	new_text = ''.join(new_text)
	pattern = re.compile(r'\b5')
	new_text = pattern.sub('$', new_text)
	return new_text
 
cnvrt(text)

#27 Python's random module includes a fxn choice() which randomly chooses
#an item from a sequence e.g. choice("aehh ") will produce 1 of 4 possible characters with the 
#letter h being twice as freq as others. Write a generator expr that produces a sequence of 
#500 randomly chosen letters drawn from "aehh " and put this expr inside a call to the ''.join()
#fxn to concatenate them into 1 long string. You should get a result that looks like 
#uncontrolled sneezing or maniacal laughter. Use split() and join() again to normalize
#the whitespace string.

from random import choice

generator = ''.join(list((choice("aehh ") for a in range(500))))
generator = generator.split(' ')
generator = ''.join(generator)
generator

#31 define the variable 'saying' to contain the list
# ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more','is', 'said', 'than', 'done', '.']
#process the list using a for loop and store length of each word in a new list 'lengths'
#Hint; begin by assinging the empty list lengths using lengths = []. Then each
#time through the loop use append() to add another length value to the list. Now do the same thing
#using a list comprehension

saying=['After', 'all', 'is', 'said', 'and', 'done', ',', 'more','is', 'said', 'than', 'done', '.']

def lengths(string):
    lengths=[]
    for a in string:
        lengths.append(len(a))
    return lengths

lengths(saying)    
    
lengths=[len(a) for a in saying]

#32 define a variable silly to contain the string 'newly formed bland ideas are inexpressible in an infuriating way'. Now write code to:
#a. split silly into a list of strings, one per word, using split() and save to a var called bland
#b. extract 2nd letter of each word in silly and join them into a string to get 'eoldrnnnna'
#c. combine words in bland back into a single string using join(). Make sur words in string are separated with white space
#d. print the words of silly in alpha order, one per line

#a
silly = 'newly formed bland ideas are inexpressible in an infuriating way'
bland = silly.split(' ')
#b
words=''
for a in bland:
    words += a[1]
print (words)
#c
bland2=' '.join(bland)
#d
srtd=sorted(bland)
for a in srtd:
    print(a)


