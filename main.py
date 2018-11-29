
from word2vec import use_w2v
#import make_prob as mp
from make_tests import a_dac_test as dac, a_definition_test as dt, a_example_test as et, a_sentence_test as st, a_w2v_test as wt

#input_model_path = 'word2vec/model/GoogleNews-vectors-negative300.bin' #0506모델에서는 sg과 cbow가 바뀜
input_model_path = 'word2vec/model/w2v_0511_cbow'
model = use_w2v.w2v()
model.load_model(input_model_path)
list_of_examples1 = [
					"big bigger ugly",
					"bad worse big",
					"bad worse bright",
					"bad worse cheap",
					"bad worse cold",
					"bad worse cool",
	                "bad worse deep",
					"bad worse easy",
					"bad worse fast",
					"bad worse good",
					"boy girl brother ",
					"boy girl brothers ",
					"boy girl dad ",
					"boy girl father",
					"boy girl grandfather",
					"boy girl grandpa ",
					"boy girl grandson ",
					"boy girl groom ",
					"boy girl he ",
					"boy girl his"
					]
# list_of_examples2 = [
# 					"code coding dance ",
# 					"code coding debug ",
# 					"code coding decrease ",
# 					"code coding describe ",
# 					"code coding discover ",
# 					"code coding enhance ",
# 					"code coding fly ",
# 					"code coding generate ",
# 					"code coding go ",
# 					"code coding implement "
# 					]
#mp.w2v_test(model, list_of_examples1)
# mp.syntax_test(model, list_of_examples2)
sentence = "The small community project was so successful that it expanded to include other communities around the country."
#sentence = According to the New York-based think tank Even B. Donaldson Adoption Institute, this preference largely has to do with who usually becomes the main caregiver of the adopted children."
st.sentence_test(model,  sentence)
#
word = "expanded" #implication
dt.definition_test(model, word)
et.example_test(model, word)
dac.definition_as_choice_test(model, word)
wt.w2v_test(model, list_of_examples1)
