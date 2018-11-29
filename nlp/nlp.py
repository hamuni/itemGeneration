import nltk
import os
import re

"""

    This NLP model is for preprocessing of texts.
    build word list, filter some inappropriate words,
    pos taggin etc.

    Basic vocabulary data would be placed in 'data' directory.

"""

def word_dictionary():
    stemmer = nltk.PorterStemmer()
    word_list = {}
    f = open("/home/yoonhee/Desktop/IGproject/voca.txt", 'r')
    words = f.readlines()
    for word in words:
        new_word = re.sub("\n","",word)
        if(new_word not in word_list and len(new_word)>0):
            word_list[word.strip()] = stemmer.stem(new_word)
    f.close()
    return word_list


def possible_blank_position(sentence,word_list):

    stop_words = set(nltk.corpus.stopwords.words('english'))

    tokenized_sentence = nltk.word_tokenize(sentence)

    without_stopwords = []

    for words in tokenized_sentence:
        if words not in stop_words and len(words) > 2:
            without_stopwords.append(words)

    stemmer = nltk.PorterStemmer()

    possible_pos_num = []

    for pos_num in range(len(tokenized_sentence)):

        stemmed_word = stemmer.stem(tokenized_sentence[pos_num])

        if stemmed_word in word_list:

            possible_pos_num.append(pos_num)

    return tokenized_sentence,without_stopwords,possible_pos_num


class NLP():

    def __init__(self,config):
        self.config = config


    def pos_tagging(self,word_list):

        result = nltk.pos_tag(word_list)

        return result

    def get_same_pos_words(self,word_list,target_word,word_dictionary):

        stemmer = nltk.PorterStemmer()

        result = []
        target_pos = self.pos_tagging([target_word])[0][1]
        pos_tagged = self.pos_tagging(word_list)

        for word_tuple in pos_tagged:
            stemmed = stemmer.stem(word_tuple[0])
            if word_tuple[1] == target_pos:
                if stemmed in word_dictionary:
                    result.append(word_tuple[0])
        return result
