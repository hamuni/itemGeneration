from nltk.corpus import wordnet as wn


def get_definition(word):
    #단어의 정의 문장들을 리스트에 넣어서 리턴
    sset = wn.synsets(word)
    meaning_lst = []
    for word in sset:
        w = word.name()
        meaning_lst.append(wn.synset(w).definition())
    return meaning_lst

def get_example(word):
    sset = wn.synsets(word)
    example_lst = []
    for word in sset:
        w = word.name()
        example_lst.append(wn.synset(w).examples())
    return example_lst
