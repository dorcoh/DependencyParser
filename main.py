from decoder import Data
from classifier import Perceptron
from features import init_feature_functions

train_data = Data('resources/dev_10.labeled')
test_data = Data('resources/dev_10.labeled')
filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0,
    }
#    'child_word_pos': 0,
#    'child_word': 0,
#    'child_pos': 0,
#    # bigram
#    'parent_pos_child_word_pos': 0,
#    'parent_word_pos_child_pos': 0,
#    'parent_pos_child_pos': 0
#}

ground_graphs = {}
gold_graph = {}

for sentence_idx, sentence in enumerate(test_data):
    for word_idx, word in sentence.items():
        if word_idx == 0:
            ground_graphs[sentence_idx] = {}
            ground_graphs[sentence_idx][0] = []
        if word[3] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[3]] = []
        if word[0] not in ground_graphs[sentence_idx].keys():
            ground_graphs[sentence_idx][word[0]] = []
        if word_idx != 0:
            ground_graphs[sentence_idx][word[3]].append(word[0])


clf = Perceptron(train_data, filter_dict=filter_dict)
clf.fit(num_iter=1)
y_pred = clf.predict(test_data)
print(clf.get_accuracy(y_pred, ground_graphs))
