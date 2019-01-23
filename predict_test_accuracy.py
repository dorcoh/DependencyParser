from parser.decoder import Data
from parser.common import pickle_load
from parser.classifier import Perceptron

comp_file = Data('resources/test.labeled', comp=False)
f = pickle_load('features.pickle')
w = pickle_load('w.pickle')

clf = Perceptron(features_tuple_pick=f, test_data=comp_file, comp=False, w=w)
y_pred = clf.predict(comp_file)
y_true = clf.ground_graph_test(comp_file)
print(clf.get_accuracy(y_pred,y_true))

# for i, pred in enumerate(y_pred):
#     print("PRED")
#     print(pred.successors)
#     print("TRUE")
#     print(y_true[i])
