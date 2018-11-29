import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import models

class w2v_model():

    def __init__(self,config=None):

        if(config is not None):
            self.config      = config

        self.model       = models.word2vec.Word2Vec(min_count=1)
        self.model_check = False
        self.sentences   = None
        self.epoch       = 20

    def load_sentence(self,input_file_name):
        self.sentences = models.word2vec.LineSentence(input_file_name)

    def build_vocab(self):
        self.model.build_vocab(self.sentences)

    def train(self,input_file_name):

        if(not(self.model_check)):
            print("load the model first.")
            return

        self.sentences = models.word2vec.LineSentence(input_file_name)

        self.model.train(
            self.sentences,
            total_examples=self.model.corpus_count,
            epochs=self.epoch)

    def save_model(self,output_file_name):
        self.model.save(output_file_name)

    def make_model(self,input_file_name,output_file_name):

        self.load_sentence(input_file_name)
        self.build_vocab()
        self.train()
        self.save_model(output_file_name)
        self.model_check = True

    def load_model(self,input_path=None):
        if(input_path == None):
            self.model = models.word2vec.Word2Vec.load(self.config.word2vec_model)
        else:
            self.model = models.word2vec.Word2Vec.load(input_path)

        self.model_check = True

    def most_similar(self,positive_words,negative_words,count):

        if not self.model_check:
            print ("Model is not checked, make or load the model")
            return False

        return self.model.wv.most_similar(positive_words,negative_words,count)

    def vocab_list(self):
        return self.model.wv.vocab

    def input_refine(self,word_list):

        vocab_list = self.vocab_list()
        return_value = []

        for word in word_list:
            if word in vocab_list:
                return_value.append(word)

        return return_value

    def get_simliar_words(self,positive_words,negative_words,type):

        positive_words_refined = self.input_refine(positive_words)
        negative_words_refined = self.input_refine(negative_words)

        result = self.most_similar(
            positive_words_refined,
            negative_words_refined,
            500)

        if(type == "word-only"):
            result = self.word_only(result)
        return result

    def w2v_train_data(self):

        input_text  = self.config.global_text
        output_text = self.config.word2vec_model

        w2v_model   = w2v_train.w2v_model()

        w2v_model.make_model_and_save(input_text,output_text)

        return

    def word_only(self,words_list):
        """
            remove similarity score.
            arg_list     :  [(word,score),(word,score), ...]
            return value :  [word,word, ...]
        """
        result = []
        for word_tuple in words_list:
            result.append(word_tuple[0])
        return result
