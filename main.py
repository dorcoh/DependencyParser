from decoder import Data
from classifier import Perceptron

data = Data('resources/train.labeled')

# baseline
filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0,
    'child_pos': 0,
    'parent_pos_child_word_pos': 0,
    'parent_word_pos_child_pos': 0,
    'parent_child_pos': 0,
}

# competition model
filter_dict_model = {
    # unigram
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0,
    'child_pos': 0,
    # bigram
    'parent_child_word_pos': 0,
    'parent_pos_child_word_pos': 0,
    'parent_word_child_word_pos': 0,
    'parent_word_pos_child_pos': 0,
    'parent_word_pos_child_word': 0,
    'parent_child_word': 0,
    'parent_child_pos': 0,
    # extra
    'parent_child_pos_distance': 0,
    'parent_child_word_distance': 0
}

clf = Perceptron(data, filter_dict_model)
clf.fit(num_iter=20)
