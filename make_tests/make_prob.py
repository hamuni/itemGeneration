import distractor as di
import logging, re, nltk
from word2vec import use_w2v
from nlp import nlp
import definition as de
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement

#lemma = WordNetLemmatizer().lemmatize(word, nltk.pos_tag(word)[0][1])
def get_lemma(words):
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(words)
    lemma_lst =[]
    for w, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lemmatizer(w)
        else:
            lemma = lemmatizer.lemmatize(w, pos = wntag)
        lemma_lst.append(lemma)
    return lemma_lst


def w2v_test(model, list_of_examples):
    for example in list_of_examples:
        i = 0
        a,b,c = example.split()
        d = model.predict_word(a,b,c)[0][0]
        print("Q.%s:%s = %s: _______" % (a,b,c))
        di.generate_distractor(model, d)
        # for w in di.generate_w2v_distractor(model, d):
        #     i = i + 1
        #     print("%d)" %i, w)
        # print("A. %s" %d)

def syntax_test(model, list_of_examples):
    for example in list_of_examples:
        i = 0
        a,b,c = example.split()
        d = model.predict_word(a,b,c)[0][0]
        print("Q.%s:%s = %s: _______" % (a,b,c))
        di.generate_distractor(model, d)

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
        # for w in di.generate_distractor(model, word):
        #     j = j +  1
        #     print("%d)" %j, w)
        # print("A. %s" %word)
        tmp[pos] = word

def definition_test(model, word):
    definition = de.get_definition(word)
    #definition 수를 변화시킬 수 있음
    print("Q.", definition[0])
    i = 0
    di.generate_distractor(model, word)
    # for w in di.generate_distractor(model, word):
    #     i = i +  1
    #     print("%d)" %i, w)
    # print("A. %s" %word)


def example_test(model, word):
    blank = "_______"
    example = de.get_example(word)
    #example의 수를 변화시킬 수 있음
    ex = example[0][0].replace(word, blank)
    print("Q.", ex)
    i = 0
    di.generate_distractor(model, word)
    # for w in di.generate_distractor(model, word):
    #     i = i +  1
    #     print("%d)" %i, w)
    # print("A. %s" %word)

#같은 lemma는 없애기
def definition_as_choice(model, word):
    lemma = get_lemma([word])
    semantic_distractor = list(set(di.semantic_distractor_de(model, word)))
    synonym = list(set(di.synonym_distractor(word)))
    synonym_lemma = get_lemma(synonym)
    semantic_lemma = get_lemma(semantic_distractor)
    print("semantic_distractor:" , semantic_distractor)
    print("synonym:" , synonym)
    distractor = set(semantic_lemma) - set(synonym_lemma) - set(lemma)
    print("distractor = semantic_distractor - (synonym_lemma + lemma)")
    print("distractor:", distractor)
    print("Q.", word)
    print("answer:", de.get_definition(word)[0])
    i = 1
    for word in distractor:
        definition = de.get_definition(word)
        print(i,".", definition[0], "\nword:", word)
        i += 1
