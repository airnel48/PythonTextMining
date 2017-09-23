# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 09:40:44 2017

@author: Eric
"""

#15 Write programs to process the Brown Corpus and find answers to the following questions:
  
import nltk
from nltk.corpus import brown
tag_words = brown.tagged_words()
wordfreq=nltk.ConditionalFreqDist(tag_words)  

 #15a Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
cond = wordfreq.conditions()
for a in cond:
    if wordfreq[a]['NNS'] > wordfreq[a]['NN']:
        print(a)

        
 #15b Which word has the greatest number of distinct tags. What are they, and what do they represent?
from operator import itemgetter
dist_tags = []
for a in cond:
	dist_tags.append((a, len(wordfreq[a])))
top_dist_tag = list(reversed(sorted(dist_tags,key=itemgetter(1))))[0]
wordfreq['that']
#the word 'that' has the highest number of distinct tags at 12. They are:
#CS (conjuction), CS-HL(conjuction), CS-NC(conjuction), DT(determiner), DT-NC(determiner-conjunction), NIL-(none), QL-(adverb), WPO(pronoun), WPO-NC(pronoun), WPS(pronoun), WPS-HL(pronoun), WPS-NC(pronoun)
 
 #15c List tags in order of decreasing frequency. What do the 20 most frequent tags represent?

tag_20 = nltk.FreqDist([b for a, b in tag_words]).most_common(20)
#the 20 most frequent tags represent NN(noun), IN(preposition), AT(definite article), JJ(adjective), .(period), ,(comma), NNS(plural noun), CC(conjunction), RB(adverb), NP(proper noun), VB(verb), VBN(past participle verb), VBD(past tense verb), CS(conjunction), PPS(pronoun), VBG(gerunds), PP$(pronoun), TO(particle), PPSS(possessive), CD(cardinal number)
 
 #15d Which tags are nouns most commonly found after? What do these tags represent?

most_common=nltk.FreqDist([a[1] for (a, b) in nltk.bigrams(tag_words) if b[1] == 'NN']).most_common(10)
#Nouns are most commonly found after AT(determiner), JJ(adjective), IN(preposition), NN(noun), PP$(pronoun), DT(determiner), CC(conjunction), VBG(gerunds) , AP(adjective), and ,(comma)

 
#18 Generate some statistics for tagged data to answer the following questions:
wordfreq=nltk.ConditionalFreqDist(tag_words) 
cond = wordfreq.conditions()   

 #18a What proportion of word types are always assigned the same part-of-speech tag?
solo= [a for a in cond if len(wordfreq[a]) == 1]
pct_solo = len(solo) / len(cond) 

 #18b How many words are ambiguous, in the sense that they appear with at least two tags?
non_solo = len(cond) - len(solo) 
 
 #15c What percentage of word tokens in the Brown Corpus involve these ambiguous words?
brown = set(brown.words())
brown_solo = [a for a in solo if a in brown]
pct_brown_non_solo = 1- (len(brown_solo) / len(brown))
 
 
#21 In 3.1 we saw a table involving frequency counts for the verbs adore, love, like, prefer and 
#preceding qualifiers absolutely and definitely. Investigate the full range of adverbs that appear before these four verbs.
tag_bigs = nltk.bigrams(tag_words)
verb_preceders = [a[0] for (a, b) in tag_bigs if b[0] in ('adore', 'love' , 'like' , 'prefer') and a[1] == ('RB')]
set(verb_preceders)

#22 We defined the regexp_tagger that can be used as a fall-back tagger for unknown words. This tagger only checks for cardinal numbers. 
#By testing for particular prefix or suffix strings, it should be possible to guess other tags. For example, we could tag any word that ends with -s as a plural noun. 
#Define a regular expression tagger (using  RegexpTagger()) that tests for at least five other patterns in the spelling of words. (Use inline documentation to explain the rules.)
patterns = [
    (r'.*ing$', 'VBG'), # gerunds
    (r'.*ed$', 'VBD'), # simple past
    (r'.*es$', 'VBZ'), # 3rd singular present
    (r'.*ould$', 'MD'), # modals
    (r'.*\'s$', 'NN$'), #possessive nouns
    (r'.*s$', 'NNS'), # plural nouns
    (r'.*', 'NN') #nouns default
]

print(nltk.RegexpTagger(patterns).tag(brown.sents()[1]))

#26 4.1 plotted a curve showing change in the performance of a lookup tagger as the model size was increased. Plot the performance curve for a unigram tagger, as the amount of training data is varied.
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    unigram_tagger = nltk.UnigramTagger(brown.tagged_sents(categories='news'))
    unigram_tagger.tag(brown.sents()[2007])
    return unigram_tagger.evaluate(brown.tagged_sents(categories='news'))
      
def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    training_data = 2 ** pylab.arange(15)
    perfs = [performance(cfd[:training_data], words_by_freq) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Unigram Tagger Performance with Varying Training Data')
    pylab.xlabel('Training Data')
    pylab.ylabel('Performance')
    pylab.show()


#29 Recall the example of a bigram tagger which encountered a word it hadn't seen during training, 
#and tagged the rest of the sentence as None. It is possible for a bigram tagger to fail part way through a 
#sentence even if it contains no unseen words (even if the sentence was used during training). 
#In what circumstance can this happen? Can you write a program to find some examples of this?
brown_tagged_sents = brown.tagged_sents(categories='news')
bigram_tagger = nltk.BigramTagger(brown_tagged_sents[:5])
sent = ['The','jury','was','charged','up','on','energy','drinks']
bigram_tagger.tag(sent)


#30 Preprocess the Brown News data by replacing low frequency words with UNK, but leaving the tags untouched. Now train and evaluate a bigram tagger on this data. How much does this help? 
#What is the contribution of the unigram tagger and default tagger now?
hifreq=nltk.FreqDist(brown.words(categories='news')).most_common(50)
tags=brown.tagged_sents(categories='news')
new_sents=[('UNK',tag) for (word,tag) in tags if word not in hifreq else (word,tag)]
sample=int(len(new_sents)*0.8) 
train=new_sents[:sample]
test=new_sents[sample:]
run1=nltk.DefaultTagger('NN')
run2=nltk.UnigramTagger(train,backoff=run1)
run3=nltk.BigramTagger(train,backoff=run2)


#31 Modify the program in 4.1 to use a logarithmic scale on the x-axis, by replacing  pylab.plot() with pylab.semilogx(). What do you notice about the shape of the resulting plot? 
#Does the gradient tell you anything?
def performance(cfd, wordlist):
    # takes in a conditional frequency and a wordlist
    # goes through every word iun the wordlist and returns a dictionary consisting of the word and the most frequent tag for that word.
    lt = dict((word, cfd[word].max()) for word in wordlist)
    # the baseline tagger is fed the resultant list of tagged words as a model. and given the default tagger to make everything else a noun.
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    # returns the evaluation score for the tagger
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    # pulls in a frequency distribution of all the words in the news category
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    # sequentially orders the words by frequency
    words_by_freq = [w for (w, _) in word_freqs]
    # makes a cfd based on the words and the frequency of their tags
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    # returns a list of evenly spaced numbers from 1 to two to the power of fifteen
    sizes = 2 ** pylab.arange(15)
    # for every size in that evenly spaced array, evaluate a baseline tagger based on a training set of that size. so it's plotting training models that get larger and larger.
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.semilogx(sizes, perfs, '-bo')
    # sets all of the axes
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

display()