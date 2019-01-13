from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import copy

class FeatureFunction(ABC):
    def __init__(self, name, data):
        self.feature_dict = {}
        self.name = name
        self.data = self.preprocess(data)

    def preprocess(self, data):
        for sentence in data:
            for tup in sentence:
                key = self.extract_key(tup, sentence)
                if key in self.feature_dict:
                    self.feature_dict[key] += 1
                else:
                    self.feature_dict[key] = 1

    def filter_features(self, min_count):
        to_remove = []
        for feature, count in self.feature_dict.items():
            if not count >= min_count:
                to_remove.append(feature)

        for feature in to_remove:
            del self.feature_dict[feature]

    def compute_size(self):
        """return size of features after preprocess"""
        return len(self.feature_dict.keys())

    @abstractmethod
    def extract_key(self, tup, sentence):
        """returns unique feature key as a tuple"""
        pass

    def __call__(self, **kwargs):
        """actual feature function - f(x,t) - applied per sample
        returns feature vector as data, row, col"""
        sentence = kwargs['sentence']
        idx_dict = kwargs['idx_dict']
        temp_dict = {}
        for tup in sentence:
            key = self.extract_key(tup, sentence)
            # filter out features
            if key not in self.feature_dict:
                return [], [], []

            if key in temp_dict:
                temp_dict[key] += 1
            else:
                temp_dict[key] = 1

        data, row, col = [], [], []
        for key, value in temp_dict.items():
            row.append(0)
            col.append(idx_dict[key])
            data.append(value)

        del temp_dict
        return data, row, col

    def get_enabled_feature(self, sentence, child, parent_id, idx_dict):
        """Returns enable feature id for (parent, child), if empty returns None"""
        child = copy.copy(child)
        child[3] = parent_id
        key = self.extract_key(child, sentence)
        if not key in self.feature_dict:
            return None
        return idx_dict[key]


class ParentWordPos(FeatureFunction):

    def extract_key(self, tup, sentence):
        parent_id = int(tup[3])
        parent_word = sentence[parent_id][1]
        parent_pos = sentence[parent_id][2]
        key = (parent_word, parent_pos)
        return key


class ParentWord(FeatureFunction):

    def extract_key(self, tup, sentence):
        parent_id  = int(tup[3])
        parent_word = sentence[parent_id][1]
        key = (parent_word)
        return key


class ParentPos(FeatureFunction):

    def extract_key(self, tup, sentence):
        parent_id = int(tup[3])
        parent_pos = sentence[parent_id][2]
        key = (parent_pos)
        return key


class ChildWordPos(FeatureFunction):

    def extract_key(self, tup, sentence):
        child_word = tup[1]
        child_pos = tup[2]
        key = (child_word, child_pos)
        return key


class ChildWord(FeatureFunction):

    def extract_key(self, tup, sentence):
        child_word = tup[1]
        key = (child_word)
        return key


def init_feature_functions(train_data, filter_dict):
    # init all functions
    callables_dict = {
        'parent_word_pos': ParentWordPos,
        'parent_word': ParentWord
    }
    for name, callable in callables_dict.items():
        callables_dict[name] = callable(name, train_data)
        callables_dict[name].filter_features(filter_dict[name])

    idx_dic = {}
    tmp_max = 0
    # build feature mapping
    for name, c in callables_dict.items():
        for k, v in c.feature_dict.items():
            idx_dic[k] = tmp_max
            tmp_max += 1

    return callables_dict, idx_dic


def compute_features_size(callables_dict):
    m = 0
    for name, feature_function in callables_dict.items():
        m += feature_function.compute_size()
        print("feature function:" + name)
        print("m: %d" % m)

    return m


def get_edge_features(sentence, child, parent_id, callables_dict, idx_dict):
    feature_indices = []
    for name, feature_function in callables_dict.items():
        feature_id = feature_function.get_enabled_feature(sentence, child, parent_id, idx_dict)
        if feature_id:
            feature_indices.append(feature_id)

    return feature_indices


if __name__ == '__main__':
    # test
    from decoder import Data

    data = Data('resources/test.labeled')
    filter_dict = {
        'parent_word_pos': 3,
        'parent_word': 3
    }
    callables_dict, idx_dict = init_feature_functions(data, filter_dict)
    m = compute_features_size(callables_dict)
    for sent in data:
        data, row, col = [], [], []
        # test for one child,parent
        features_ids = get_edge_features(sent, sent[0], 4, callables_dict, idx_dict)
        for name, feature_function in callables_dict.items():
            data_tmp, row_tmp, col_tmp = feature_function(sentence=sent, idx_dict=idx_dict)
            data += data_tmp
            row += row_tmp
            col += col_tmp

        sentence_feature_vector = csr_matrix((data, (row, col)), shape=(1, m))

        print(sentence_feature_vector)