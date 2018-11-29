from . import distractor as di
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def syntax_test(model, list_of_examples):
    for example in list_of_examples:
        i = 0
        a,b,c = example.split()
        d = model.predict_word(a,b,c)[0][0]
        print("Q.%s:%s = %s: _______" % (a,b,c))
        di.generate_distractor(model, d)
