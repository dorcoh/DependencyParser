from abc import ABC, abstractmethod
from copy import copy
import operator

# feature templates decorators


def optional(method):
    def wrapped(*args, **kw):
        try:
            result = method(*args, **kw)
            return result
        except:
            return None

    return wrapped


def multiple(key_in, key_out):
    """key_in is key of comp, key_out format is e.g., parent_pos, child_word (must have _identifier at end)"""
    def decorator(method):
        def wrapped(*args, **kw):
            if key_in in args[1]:
                for elem in args[1][key_in]:
                    pos_or_word = key_out.split('_')[-1]
                    idx = 2 if pos_or_word == 'pos' else 1
                    args[1][key_out] = elem[idx]
                    # add directions
                    cid = args[1]['child_id']
                    pid = args[1]['parent_id']
                    sid = int(elem[0])
                    pc_dir = 1 if pid - cid > 0 else 0
                    cs_dir = 1 if cid - sid > 0 else 0
                    args[1]['direction'] = (pc_dir, cs_dir)
                if args[1][key_in]:
                    return method(*args, **kw)
            return None
        return wrapped
    return decorator


class FeatureFunction(ABC):
    def __init__(self, name, data, baseline):
        self.feature_dict = {}
        self.name = name
        self.baseline = baseline
        self.preprocess(data)

    def parse(self, tup, sentence):
        """parse current sample feature components into shared dict"""
        parent_id = int(tup[3])
        child_id = int(tup[0])
        parent_word = sentence[parent_id][1]
        parent_pos = sentence[parent_id][2]
        child_word = tup[1]
        child_pos = tup[2]
        direction = 1 if parent_id - child_id > 0 else 0

        comp = {
            'parent_id': parent_id,
            'child_id': child_id,
            'parent_word': parent_word,
            'parent_pos': parent_pos,
            'child_word': child_word,
            'child_pos': child_pos,
            'direction': direction
        }

        # optional
        if child_id < len(sentence) - 1:
            comp['n_child_pos'] = sentence[child_id+1][2]

        if child_id > 1:
            comp['p_child_pos'] = sentence[child_id-1][2]

        if parent_id < len(sentence) - 1:
            comp['n_parent_pos'] = sentence[parent_id+1][3]

        if parent_id > 1:
            comp['p_parent_pos'] = sentence[parent_id-1][3]

        # check for child of child or sibling
        childs = []
        for idx, word in sentence.items():
            if idx == 0:
                continue
            if word[3] == child_id or word[3] == parent_id:  # child's child or sibling
                childs.append(word)
        if childs:
            comp['childs'] = childs

        return comp

    def preprocess(self, data):
        for sentence in data:
            for i in range(1, len(sentence.keys())):
                comp = self.parse(sentence[i], sentence)
                key = self.extract_key(comp)
                if key is None:
                    continue
                if key in self.feature_dict:
                    self.feature_dict[key] += 1
                else:
                    self.feature_dict[key] = 1

    def filter_features(self, min_count):
        to_remove = []
        for feature, count in self.feature_dict.items():
            if not count > min_count:
                    to_remove.append(feature)

        for feature in to_remove:
            del self.feature_dict[feature]

    def compute_size(self):
        """return size of features after preprocess"""
        return len(self.feature_dict.keys())

    @abstractmethod
    def extract_key(self, comp):
        """returns unique feature key as a tuple
        key has a form (feature_id, attr1, attr2..)"""
        pass

    def __call__(self, **kwargs):
        """actual feature function - f(x,t) - applied per sample
        returns feature vector as data, row, col"""
        sentence = kwargs['sentence']
        idx_dict = kwargs['idx_dict']
        temp_dict = {}
        for tup in sentence:
            comp = self.parse(tup, sentence)
            key = self.extract_key(comp)
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
        child = copy(child)
        child[3] = parent_id
        comp = self.parse(child, sentence)
        key = self.extract_key(comp)
        if not key in self.feature_dict:
            return None
        return idx_dict[key]


class ParentWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (1, c['parent_word'], c['parent_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWord(FeatureFunction):

    def extract_key(self, c):
        key = (2, c['parent_word'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ParentPos(FeatureFunction):

    def extract_key(self, c):
        key = (3, c['parent_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (4, c['child_word'], c['child_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (5, c['child_word'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (6, c['child_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key

# bigram


class ParentChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (7, c['parent_word'], c['parent_pos'], c['child_word'], c['child_pos'], c['direction'])
        return key


class ParentPosChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (8, c['parent_pos'], c['child_word'], c['child_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWordChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (9, c['parent_word'], c['child_word'], c['child_pos'], c['direction'])
        return key


class ParentWordPosChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (10, c['parent_word'], c['parent_pos'], c['child_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWordPosChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (11, c['parent_word'], c['parent_pos'], c['child_word'], c['direction'])
        return key


class ParentChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (12, c['parent_word'], c['child_word'], c['direction'])
        return key


class ParentChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (13, c['parent_pos'], c['child_pos'], c['direction'])
        if self.baseline:
            return key[:-1]
        return key


# optional (could be None)
class PosNeighA(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (14, c['parent_pos'], c['n_parent_pos'], c['p_child_pos'], c['child_pos'], c['direction'])
        return key


class PosNeighB(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (15, c['p_parent_pos'], c['parent_pos'], c['p_child_pos'], c['child_pos'], c['direction'])
        return key


class PosNeighC(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (16, c['parent_pos'], c['n_parent_pos'], c['child_pos'], c['n_child_pos'], c['direction'])
        return key


class PosNeighD(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (17, c['p_parent_pos'], c['parent_pos'], c['child_pos'], c['n_child_pos'], c['direction'])
        return key


class PosParentChildSibling(FeatureFunction):

    @multiple('childs', 'child_child_pos')
    def extract_key(self, c):
        key = (18, c['parent_pos'], c['child_pos'], c['child_child_pos'], c['direction'])
        return key


class WordParentChildSibling(FeatureFunction):

    @multiple('childs', 'child_child_word')
    def extract_key(self, c):
        key = (19, c['parent_word'], c['child_word'], c['child_child_word'], c['direction'])
        return key


feature_functions = {
    # unigram
    'parent_word_pos': ParentWordPos,
    'parent_word': ParentWord,
    'parent_pos': ParentPos,
    'child_word_pos': ChildWordPos,
    'child_word': ChildWord,
    'child_pos': ChildPos,
    # bigram
    'parent_child_word_pos': ParentChildWordPos,
    'parent_pos_child_word_pos': ParentPosChildWordPos,
    'parent_word_child_word_pos': ParentWordChildWordPos,
    'parent_word_pos_child_pos': ParentWordPosChildPos,
    'parent_word_pos_child_word': ParentWordPosChildWord,
    'parent_child_word': ParentChildWord,
    'parent_child_pos': ParentChildPos,
    # extra
    'pos_next_parent_previous_child': PosNeighA,
    'pos_previous_parent_previous_child': PosNeighB,
    'pos_next_parent_next_child': PosNeighC,
    'pos_previous_parent_next_child': PosNeighD,
    'pos_parent_child_sibling': PosParentChildSibling,
    'word_parent_child_sibling': WordParentChildSibling
}


def init_feature_functions(train_data, filter_dict, baseline):
    callables_dict = {}
    feature_counts = {}
    for name in filter_dict.keys():
        if name == 'parent_word_pos':
            pass
        callables_dict[name] = feature_functions[name](name, train_data, baseline)
        callables_dict[name].filter_features(filter_dict[name])
        # add feature dicts
        feature_counts = {**callables_dict[name].feature_dict, **feature_counts}

    idx_dic = {}
    tmp_max = 0
    # build feature mapping
    for name, c in callables_dict.items():
        for k, v in c.feature_dict.items():
            idx_dic[k] = tmp_max
            tmp_max += 1

    return callables_dict, idx_dic, feature_counts


def compute_features_size(callables_dict):
    m = 0
    for name, feature_function in callables_dict.items():
        curr_m = feature_function.compute_size()
        print("feature function:" + name)
        print("m: %d" % curr_m)
        m += curr_m
    print("total features: %d" % m)
    return m


def get_features(sentence, child, parent_id, callables_dict, idx_dict):
    feature_indices = []
    for name, feature_function in callables_dict.items():
        feature_id = feature_function.get_enabled_feature(sentence, child, parent_id, idx_dict)
        if feature_id:
            feature_indices.append(feature_id)

    return feature_indices


def debug_features(callables_dict, idx_dict, w, feature_counts):
    print("Debugging features")
    s = ""
    for i, elem in enumerate(w):
        if elem == 0:
            rev_idx_dict = dict((v, k) for k, v in idx_dict.items())
            s += "Feature " + str(rev_idx_dict[i]) + " , count: " + str(feature_counts[rev_idx_dict[i]]) + "\n"
    with open("debug_features.log", 'w') as handle:
        handle.write(s)
    # for name, feature_functions in callables_dict.items():
    #     print("10% lowest features counts in " + name)
    #     last_idx = int(len(feature_functions.feature_dict.items()) / 10)
    #     items = sorted(feature_functions.feature_dict.items(), key=operator.itemgetter(1), reverse=False)[:last_idx]
    #     print(items)
