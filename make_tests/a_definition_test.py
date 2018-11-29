from . import distractor as di
from . import definition as de
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def definition_test(model, word):
    definition = de.get_definition(word)
    #definition 수를 변화시킬 수 있음
    print("Q.", definition[0])
    i = 0
    di.generate_distractor(model, word)
