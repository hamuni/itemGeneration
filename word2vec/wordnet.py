from nltk.corpus import wordnet as wn


def get_meaning(word):
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

def get_hypo(word):
    #단어의 하의어를 리스트에 넣음
    sset = wn.synsets(word)[0]
    w = wn.synset(sset.name())
    hypo_lst = []
    for hypo in w.hyponyms():
        h = hypo.name()
        hypo_lst.append(h)
    return hypo_lst

def get_hyper(word):
    #단어의 상위어를 리스트에 넣음
    sset = wn.synsets(word)[0]
    w = wn.synset(sset.name())
    hyper_lst = []
    for hyper in w.hypernyms():
        h = hyper.name()
        hyper_lst.append(h)
    return hyper_lst

def get_synsets(word):
    sset = wn.synsets(word)
    syn_lst = []
    for word in sset:
        w = word.name()
        syn_lst.append(w)
    return syn_lst

#derivationally_related_forms and pertainyms 사용할지 결정

def get_antonyms(word):
    sset = wn.synsets(word)[0]
    w = wn.synset(sset.name())
    #print(w.antonyms())
    ant_lst = []
    for ant in w.lemmas()[0].antonyms():
        a = ant.name()
        ant_lst.append(a)
    return ant_lst

# similarity는 word2vec 사용 후에 쓰기 -> most_similar 에 대한 것은 없음
# def get_similar(word):
#     sset = wn.synsets(word)[0]
#     similarity =


if __name__ == "__main__":
    #get_meaning("confusion")
    #get_synsets("confusion")
    #get_antonyms("beautiful")
    print(get_meaning("beautiful"))
    print(get_example("beautiful"))
