from parser.decoder import Data
from parser.common import pickle_load, tag_comp
from parser.classifier import Perceptron
from parser.common import timeit

@timeit
def tag():
    comp_file = Data('resources/comp.unlabeled', comp=True)
    f = pickle_load('features.pickle')
    w = pickle_load('w.pickle')

    clf = Perceptron(features_tuple_pick=f, test_data=comp_file, comp=True, w=w)
    y_pred = clf.predict(comp_file)
    tag_comp(y_pred, comp_file, 'comp.labeled')

tag()