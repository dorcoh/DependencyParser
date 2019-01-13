from decoder import Data
from classifier import Perceptron
from features import init_feature_functions

bla = Data('resources/test.labeled')
filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0
}

ground_graphs = {}

for sentence_idx, sentence in enumerate(bla):
    for word_idx, word in sentence.items():
        if word_idx == 0:
            ground_graphs[sentence_idx] = {}
        if word[3] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[3]] = []
        if word[0] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[0]] = []

        ground_graphs[sentence_idx][word[3]].append(word[0])


clf = Perceptron(bla, ground_graphs, filter_dict)
clf.fit()
