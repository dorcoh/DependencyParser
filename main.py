import sys
from parser.decoder import Data
from parser.common import tag_comp
from parser.classifier import Perceptron
from parser.features_params import filter_dict, filter_dict_model


def main(argv):
    # read params
    if argv:
        num_iter = int(argv[0])
        is_baseline = bool(int(argv[1]))
        early_stopping = bool(int(argv[2]))
        model_name = str(argv[3])
    else:
        print("Usage: python main.py num_iter is_baseline(1/0) early_stopping(1/0) model_name")
        return

    if is_baseline:
        feature_dict = filter_dict
    else:
        feature_dict = filter_dict_model

    train_data = Data('resources/dev_20.labeled')
    test_data = Data('resources/dev_20.labeled')

    clf = Perceptron(train_data=train_data, test_data=test_data, filter_dict=feature_dict, baseline=is_baseline,
                     early_stopping=early_stopping, model_name=model_name)
    clf.fit(num_iter=num_iter, debug=False)
    y_pred = clf.predict(test_data)
    tag_comp(y_pred, test_data, 'ItaiTag.labeled')


if __name__ == '__main__':
    main(sys.argv[1:])
