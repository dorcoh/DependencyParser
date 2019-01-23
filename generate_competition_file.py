from parser.decoder import Data
from parser.common import pickle_load, tag_comp
from parser.classifier import Perceptron
from parser.common import timeit

@timeit
def tag():
    comp_file = Data('resources/comp.unlabeled', comp=True)
    f_baseline = pickle_load('features-baseline.pickle')
    w_baseline = pickle_load('w-baseline.pickle')
    name_baseline = 'comp_m1_301216321.labeled'
    f_model = pickle_load('features-model.pickle')
    w_model = pickle_load('w-model.pickle')
    name_model = 'comp_m2_301216321.labeled'
    params = [(f_baseline, w_baseline, name_baseline), (f_model, w_model, name_model)]

    for p in params:
        f, w, name = p[0], p[1], p[2]
        clf = Perceptron(features_tuple_pick=f, test_data=comp_file, comp=True, w=w)
        y_pred = clf.predict(comp_file)
        tag_comp(y_pred, comp_file, name)

tag()