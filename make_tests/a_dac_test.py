
from . import distractor as di
from . import definition as de
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import logging, nltk

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def definition_as_choice_test(model, word):
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
