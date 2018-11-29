from gensim import models
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class w2v():
    def __init__(self):
        self.model = None #model

    def load_model(self, input_file):
        self.model = models.word2vec.Word2Vec.load(input_file)
        #self.model = models.KeyedVectors.load_word2vec_format(input_file, binary = True, limit = 2600000)

    def most_similar(self, positive, negative, count):
        return self.model.wv.most_similar(positive,negative,count)

    def accuracy(self, eval_set):
        return self.model.accuracy(eval_set)

    def doesnt_match(self, str):
        return self.model.wv.doesnt_match(str.split())

    def similarity(self, str1, str2):
        return self.model.wv.similarity(str1, str2)

    def predict_output(self, context_words_list, topn):
        return self.model.predict_output_word(context_words_list, topn=topn)

    def predict_word(self, w1, w2, w3):
        return self.model.most_similar([w3,w2], [w1], 1)



if __name__ == '__main__':

    input_model_path = './model/GoogleNews-vectors-negative300.bin' #0506모델에서는 sg과 cbow가 바뀜
    model = w2v()
    model.load_model(input_model_path)
    # print(model.most_similar(['woman', 'king'], ['man'], 2))
    # print(model.similarity('woman', 'man'))
    # print(model.most_similar(['woman', 'king'], ['man'], 2))
    # print(model.doesnt_match("breakfast cereal dinner lunch"))
    model.accuracy('questions-words.txt')
    # examples = ["he his she", "big bigger ugly", "going went being"]
    # for example in examples:
    #     a,b,x = example.split()
    #     predicted = model.predict_word(a,b,x)
    #     print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
