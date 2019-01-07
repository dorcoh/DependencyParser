from decoder import Data
from classifier import Perceptron
from features import init_feature_functions

bla = Data('resources/test.labeled')
init_feature_functions(bla)
for item in bla:
    print(item)
