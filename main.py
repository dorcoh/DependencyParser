import sys
from decoder import Data
from classifier import Perceptron
from features_params import filter_dict, filter_dict_model

def main(argv):
    # read params
    if argv:
        num_iter = int(argv[0])
        is_baseline = bool(argv[1])
    else:
        print("Usage: python main.py num_iter is_baseline(1/0)")
        return

    if is_baseline:
        feature_dict = filter_dict
    else:
        feature_dict = filter_dict_model

    train_data = Data('resources/dev_20.labeled')
    test_data = Data('resources/dev_10.labeled')

    clf = Perceptron(train_data=train_data, test_data=test_data, filter_dict=feature_dict, baseline=is_baseline)
    clf.fit(num_iter=num_iter)


if __name__ == '__main__':
    main(sys.argv[1:])