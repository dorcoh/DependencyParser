from decoder import Data
from classifier import Perceptron

data = Data('resources/dev_10.labeled')

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

clf = Perceptron(data, filter_dict)
clf.fit(num_iter=20)
