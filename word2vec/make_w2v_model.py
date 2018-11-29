import warnings, logging, os, gensim
from gensim import models
import gzip

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class w2v():
    def __init__(self):
        self.sentences = None
        self.model = None

    def save_model(self, output_file):
        self.model.save(output_file)

    def load_model(self, input_file):
        self.model = models.word2vec.Word2Vec.load(input_file)

    def make_model(self, data_path, model_path):
        self.sentence = models.word2vec.LineSentence(data_path)
        self.model = models.word2vec.Word2Vec(
                        self.sentence,
                        min_count=30,
                        window = 5,
                        size = 300, #size between 100-300, 300 for extreme accuracy
                        sg = 0, #0: cbow, 1: skipgram -CHANGE
                        workers = 10)
        self.save_model(model_path)
        #self.load_model(model_path)

    def most_similar(self, positive, negative, count):
        return self.model.wv.most_similar(positive,negative,count)

if __name__ == '__main__':

    input_path = '../crawler/wiki_article_16.txt'  #dataset
    output_path = './model/w2v_0506_sg'  #model -CHANGE
    w2v().make_model( input_path, output_path)
    model = w2v()
    model.load_model(output_path)
    print(model.most_similar(['woman', 'king'], ['man'], 1 ))
