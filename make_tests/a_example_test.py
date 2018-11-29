from . import distractor as di
from . import definition as de
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def example_test(model, word):
    blank = "_______"
    example = de.get_example(word)
    #example의 수를 변화시킬 수 있음
    ex = example[0][0].replace(word, blank)
    print("Q.", ex)
    i = 0
    di.generate_distractor(model, word)
