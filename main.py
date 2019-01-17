from decoder import Data
from classifier import Perceptron
from features import init_feature_functions

bla = Data('resources/dev_10.labeled')

filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    'child_word_pos': 0,
    'child_word': 0,
    'child_pos': 0,
    # bigram
    'parent_pos_child_word_pos': 0,
    'parent_word_pos_child_pos': 0,
    'parent_pos_child_pos': 0
}

ground_graphs = {}
gold_graph = {}

for sentence_idx, sentence in enumerate(bla):
    for word_idx, word in sentence.items():
        if word_idx == 0:
            ground_graphs[sentence_idx] = {}
            gold_graph[sentence_idx] = {}
            gold_graph[sentence_idx][0] = []
            ground_graphs[sentence_idx][0] = []
        if word[3] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[3]] = []
            gold_graph[sentence_idx][word[3]] = []
        if word[0] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[0]] = []
            gold_graph[sentence_idx][word[0]] = []
        if word_idx != 0:
            ground_graphs[sentence_idx][word[3]].append(word[0])
            gold_graph[sentence_idx][word[3]].append(word[0])
            ground_graphs[sentence_idx][0].append(word[0])


clf = Perceptron(bla, ground_graphs, filter_dict, gold_graph, num_iter=20)
clf.fit()
