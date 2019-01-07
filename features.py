from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix


class FeatureFunction(ABC):
    def __init__(self):
        self.feature_dict = {}
        self.name = None

    def compute_size(self):
        """should fill the dictionary before calling this"""
        return len(self.feature_dict.keys())

    @abstractmethod
    def __call__(self, **kwargs):
        """actual feature function - f(x,t) - applied per sample"""
        pass


class ParentWordPos(FeatureFunction):
    def __init__(self):
        # this call should fill the dictionary
        # compute its size
        # fill its name
        super().__init__()

    def __call__(self, **kwargs):
        # list of tuples
        sentence = kwargs['sentence']
        idx_dict = kwargs['idx_dict']
        temp_dict = {}
        row = []
        col = []
        data = []
        for tup in sentence:
            parentid = int(tup[3]) - 1
            parent_word = sentence[parentid][1]
            parent_pos = sentence[parentid][2]
            key = (parent_word, parent_pos)
            if key in temp_dict:
                temp_dict[key] += 1
            else:
                temp_dict[key] = 1

        for key, value in temp_dict.items():
            row.append(0)
            col.append(idx_dict[key])
            data.append(value)

        return data, row, col

    def fill_sentence(self, sentence):
        for tup in sentence:
            parentid = int(tup[3]) - 1
            parent_word = sentence[parentid][1]
            parent_pos = sentence[parentid][2]
            key = (parent_word, parent_pos)
            if key in self.feature_dict:
                self.feature_dict[key] += 1
            else:
                self.feature_dict[key] = 1


def init_feature_functions(data):
    # init all functions
    callables = [ParentWordPos()]
    for callable in callables:
        for sentence in data:
            callable.fill_sentence(sentence)

    idx_dic = {}
    tmp_max = 0
    # build feature mapping
    for i in callables:
        for k, v in i.feature_dict.items():
            idx_dic[k] = tmp_max
            tmp_max += 1

    return callables, idx_dic


if __name__ == '__main__':
    # test
    from decoder import Data

    data = Data('resources/test.labeled')
    callables, idx_dict = init_feature_functions(data)
    for sent in data:
        data, row, col = [], [], []
        for feature_function in callables:
            data_tmp, row_tmp, col_tmp = feature_function(sentence=sent, idx_dict=idx_dict)
            data += data_tmp
            row += row_tmp
            col += col_tmp

        m = feature_function.compute_size()
        sentence_feature_vector = csr_matrix((data, (row, col)), shape=(1, m))
