from decoder import Data
from classifier import Perceptron

train_data = Data('resources/dev_10.labeled')
test_data = Data('resources/dev_10.labeled')
filter_dict = {
    'parent_word_pos': 0,
    'parent_word': 0,
    'parent_pos': 0
}
#    'child_word_pos': 0,
#    'child_word': 0,
#    'child_pos': 0,
#    # bigram
#    'parent_pos_child_word_pos': 0,
#    'parent_word_pos_child_pos': 0,
#    'parent_pos_child_pos': 0
# }


clf = Perceptron(train_data, filter_dict=filter_dict, test_data=test_data)
clf.fit(num_iter=1)
y_pred = clf.predict(test_data)
ground_graphs = clf.ground_graph_test(test_data)
print(clf.get_accuracy(y_pred, ground_graphs))
