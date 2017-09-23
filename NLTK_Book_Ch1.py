# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#HW1

import nltk
#nltk.download()
from nltk.book import *

#18  Using list addition, and the set and sorted operations, compute the vocabulary of the sentences sent1 ... sent8.
sentences=[sent1]+[sent2]+[sent3]+[sent4]+[sent5]+[sent6]+[sent7]+[sent8]
sorted(set(sentences))

#19 What is the difference between the following two lines? Which one will give a larger value? Will this be the case for other texts?
sorted(set(w.lower() for w in text1))
#This first line sets all words to lowercase, then removes any duplicate words, 
#then sorts the words alphabetically
sorted(w.lower() for w in set(text1))
#This second line removes any duplicate words that have the same capitalization, 
#then sets all words to lowercase,
#then sorts the words by capitalization and alphabetically. 
#The second list will be longer as there will be word duplicates since we 
#only removed duplicates with the same capitalization.

#20 What is the difference between the following two tests: w.isupper() and not w.islower()?
#the difference between w.isupper and not w.islower() is that w.isupper will
#only include words with capitalization. not.islower() will include words
#that have capitalization and puncutation which is also not lower case.

#21 Write the slice expression that extracts the last two words of text2.
end=len(text2)-2
text2[:end]

#22 Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.

#find all four letter words in
text5
#use _ to show these words in decreasing order of frequency
FreqDist

flw=[w for w in text5 if len(w)=4]
distflw=set(flw)
 
fdist=FreqDist(flw)

#24 
#find all words in text6 that 
#   a. end in 'ize'
#   b. contain the letter z
#   c. Contain the sequence of letters pt
#   d. Have all lowercase letters except for an initial capital

for w in text6:
    if w.endswith('ize')
    and 'z' in w
    and 'pt' in w
    and not w.isupper()
    
#25 Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now write code to perform the following tasks:
sent=['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']

#a. print all words beginning with sh
#b. print all words longer than 4 characters

for token in sent:
    if token.startswith(sh) :
        print(token)
        
for token in sent:
    if len(token) > 4 :
        print(token)

#26 What does the following Python code do?  sum(len(w) for w in text1) Can you use it to work out the average word length of a text?


#this code counts the length of each word in text 1 and then sums them to
#obtain the total number of characters

#yes, you can get avg word length by dividing the sum by the count of words 

#27 Define a function called vocab_size(text) that has a single parameter for the text, and which returns the vocabulary size of the text.

#define fx called vocab_size(text) that has a single param for the text
#and retrns the vocab size of the text

def vocab_size(text):
    return len(text)

#28
def percent(word,text):
    return text.count(word) / len(text)


#define fxn percent(word,text) that calculates how often a given word occurs
#in a text and expresses the result as a percentage





















































