from decoder import Data
from classifier import Perceptron

train_data = Data('resources/dev_20.labeled')
test_data = Data('resources/dev_10.labeled')


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
    'pre_child_pos': 0,
    'next_child_pos': 0,
    'next_parent_pos': 0,
    'pre_parent_pos': 0
}


clf = Perceptron(train_data=train_data, test_data=test_data, filter_dict=filter_dict, baseline=False)
clf.fit(num_iter=10)
y_pred = clf.predict(test_data)
# final acc
ground_graphs = clf.ground_graph_test(test_data)
print(clf.get_accuracy(y_pred, ground_graphs))
