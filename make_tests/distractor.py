#from word2vec import use_w2v
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from word2vec import use_w2v
from nltk.corpus import wordnet as wn
import py_stringmatching as sm
import jellyfish as jf
import random
from nltk.stem.wordnet import WordNetLemmatizer

def generate_distractor(model, word):
    distractor_set = set()
    #distractor_set.update(lst)
    print("answer:", word)
    print("1.(semantic)", semantic_distractor(model, word))
    #distractor_set.update(hypernym_distractor(word))
    print("2.(shape)" ,shape_distractor(word))
    print("3.(hypernym)", hypernym_distractor(word))
    #distractor_set.update(hyponym_distractor(word))
    print("4.(hyponym)", hyponym_distractor(word))
    #distractor_set.update(synonym_distractor(word))
    #distractor_set.update(antonym_distractor(word))
    antonym_lst = list(set(antonym_distractor(word)))
    print("5.(antonym)", antonym_lst)
    if len(antonym_lst) != 0:
        first_antonym = antonym_lst[0]
        print("6.(antonym's hypernym)", hypernym_distractor(first_antonym))
        print("7.(antonym's hyponym)", hypernym_distractor(first_antonym) )
        print("8.(antonym's shape)", shape_distractor(first_antonym))
        antonym_semantic_lst = []
        antonym_lemma = WordNetLemmatizer().lemmatize(word, pos ="n")
        antonym_lemma = antonym_lemma.lower()
        for w in model.most_similar(antonym_lemma,[],5):
            antonym_semantic_lst.append(w[0])
        print("9.(antonym's semantic)", antonym_semantic_lst)
    #distractor_set.update(shape_distractor(word))
    print("* avoid synonyms:", set(synonym_distractor(word))- set(word) , "\n")


    # distractor_list = list(distractor_set)
    # random.shuffle(distractor_list)
    # return_list = distractor_list[:5]
    # if word in return_list:
    #     return return_list
    # else:
    #     del return_list[0]
    #     return_list.append(word)
    #     random.shuffle(return_list)
    #     return return_list

def generate_w2v_distractor(model, word):
    distractor_set = set()
    lst = []
    for w in model.most_similar(word,[],5):
        lst.append(w[0])
    distractor_set.update(lst)
    print("semantic: ", lst)
    #distractor_set.update(hypernym_distractor(word))
    #print("hypernym", hypernym_distractor(word))
    #distractor_set.update(hyponym_distractor(word))
    #print("hyponym", hyponym_distractor(word))
    synonym_set = set(synonym_distractor(word))
    print("synonym: ", synonym_distractor(word))
    distractor_set = distractor_set - synonym_set
    print(distractor_set)
    #문제 특성상 반의어는 필요하지 않음
    #distractor_set.update(antonym_distractor(word))
    #print("antonym", antonym_distractor(word))
    #의미가 더 중요한 문제이기 때문
    #distractor_set.update(shape_distractor(word))
    distractor_list = list(distractor_set)
    #print("distractor_list:", distractor_list)
    random.shuffle(distractor_list)
    return_list = distractor_list[:5]
    if word in return_list:
        return return_list
    else:
        del return_list[0]
        return_list.append(word)
        random.shuffle(return_list)
        return return_list

# 의미 유사도를 이용한 오답 - from word2vec
def semantic_distractor(model, word):
    semantic_lst = []
    lemma = WordNetLemmatizer().lemmatize(word, pos="n")
    lemma = lemma.lower()
    for w in model.most_similar(lemma,[],5):
        semantic_lst.append(w[0])
    return semantic_lst

def semantic_distractor_de(model, word):
    semantic_lst = []
    lemma = WordNetLemmatizer().lemmatize(word, pos="n")
    lemma = lemma.lower()
    for w in model.most_similar(lemma,[],10):
        semantic_lst.append(w[0])
    return semantic_lst

# 상위어 오답
def hypernym_distractor(word):
    if wn.synsets(word) == []:
        #print("no hypernym_distractor")
        return []
    else:
        sset = wn.synsets(word)[0]
        w = wn.synset(sset.name())
        hyper_lst = []
        for hyper in w.hypernyms():
            h = hyper.name()[:-5]
            hyper_lst.append(h)
        return hyper_lst

# 하위어 오답
def hyponym_distractor(word):
    if wn.synsets(word) == []:
        #print("no hyponym_distractor")
        return []
    else:
        sset = wn.synsets(word)[0]
        w = wn.synset(sset.name())
        hypo_lst = []
        for hypo in w.hyponyms():
            h = hypo.name()[:-5]
            hypo_lst.append(h)
        return hypo_lst[:5]

# 유의어 오답
def synonym_distractor(word):
    if wn.synsets(word) == []:
        #print("no synonym_distractor")
        return []
    else:
        sset = wn.synsets(word)
        syn_lst = []
        for word in sset:
            w = word.name()[:-5]
            syn_lst.append(w)
        return syn_lst[:5]

# 반의어 오답
def antonym_distractor(word):
    if wn.synsets(word) == []:
        #print("no antonym_distractor")
        return []
    else:
        sset = wn.synsets(word)[0]
        w = wn.synset(sset.name())
        #print(w.antonyms())
        ant_lst = []
        for ant in w.lemmas()[0].antonyms():
            a = ant.name()
            ant_lst.append(a)
        return ant_lst[:5]

#형태 유사도를 이용한 오답
def shape_distractor(word1):
    dic = open('/home/yoonhee/Desktop/IGproject/voca.txt', 'r')
    dic_line = dic.readlines()
    editex = sm.Editex()
    similarity_list = []
    list = []

    for line in dic_line:
        splited_line = line.split()
        word2 = splited_line[0].lower()

        if(word1 != word2):
            similarity = (editex.get_sim_score(word1, word2) + jf.jaro_distance(word1, word2))/2
            similarity_list.append([word2, similarity])

    #print(similarity_list)
    similarity_list.sort(key = lambda x:x[1], reverse = True)

    for item in similarity_list[:5]:
        list.append(item[0])
    #print(tuple(similarity_list[0][:5]))
    return list

# # 자주 틀리는 오답
# def frequent_error(word):
