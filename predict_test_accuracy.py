from parser.decoder import Data
from parser.common import pickle_load
from parser.classifier import Perceptron


def pred():
    comp_file = Data('resources/test.labeled', comp=False)
    f_baseline = pickle_load('features-baseline.pickle')
    w_baseline = pickle_load('w-baseline.pickle')
    f_model = pickle_load('features-model.pickle')
    w_model = pickle_load('w-model.pickle')
    params = [(f_baseline, w_baseline), (f_model, w_model)]

    for p in params:
        f, w = p[0], p[1]

        clf = Perceptron(features_tuple_pick=f, test_data=comp_file, comp=False, w=w)
        y_pred = clf.predict(comp_file)
        y_true = clf.ground_graph_test(comp_file)
        print("Accuracy:" + str(clf.get_accuracy(y_pred,y_true)))

pred()
