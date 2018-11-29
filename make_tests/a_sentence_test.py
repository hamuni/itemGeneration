import logging, re
from nlp import nlp
from . import distractor as di

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sentence_test(model, sentence):
    replaced = sentence.replace("_"," ")
    sentence = re.sub(r'[^a-zA-Z0-9?\"\'’=\.\,]',' ', replaced)
    word_dictionary = nlp.word_dictionary()
    blank = "_______"
    #voca_check
    tokenized_sentence, without_stopwords, possible_blank_pos = nlp.possible_blank_position(sentence, word_dictionary)

    #print("tokenized", tokenized_sentence, "without", without_stopwords,"possible", possible_blank_pos)
    i = 0
    for pos in possible_blank_pos:
        tmp = tokenized_sentence
        i = i + 1
        word = tokenized_sentence[pos]
        tmp[pos] = blank
        #sen = sentence.replace(word, blank)
        print("#", " ".join(tmp))
        j = 0
        di.generate_distractor(model, word)
        tmp[pos] = word
