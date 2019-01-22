from abc import ABC, abstractmethod
from collections import OrderedDict


def calc_distance(id_a, id_b):
    diff = id_a - id_b
    if diff > 3:
        distance = 4
    elif diff < -3:
        distance = -4
    else:
        distance = diff

    return distance


def optional(method):
    def wrapped(*args, **kw):
        try:
            result = method(*args, **kw)
            return result
        except:
            return None

    return wrapped


def parse(tup, sentence, baseline):
    """parse current sample feature components into shared dict"""
    parent_id = int(tup[3])
    child_id = int(tup[0])
    parent_word = sentence[parent_id][1]
    parent_pos = sentence[parent_id][2]
    child_word = tup[1]
    child_pos = tup[2]
    distance = calc_distance(parent_id, child_id)

    comp = {
        'parent_id': parent_id,
        'child_id': child_id,
        'parent_word': parent_word,
        'parent_pos': parent_pos,
        'child_word': child_word,
        'child_pos': child_pos,
        'distance': distance
    }

    if baseline:
        return comp

    # optional
    if child_id < len(sentence) - 1:
        comp['n_child_pos'] = sentence[child_id+1][2]

    if child_id > 1:
        comp['p_child_pos'] = sentence[child_id-1][2]

    if parent_id < len(sentence) - 1:
        comp['n_parent_pos'] = sentence[parent_id+1][2]

    if parent_id > 1:
        comp['p_parent_pos'] = sentence[parent_id-1][2]

    return comp


class FeatureFunction(ABC):
    def __init__(self, name, data, baseline):
        self.feature_dict = OrderedDict()
        self.name = name
        self.baseline = baseline
        self.preprocess(data)

    def preprocess(self, data):
        for sentence in data:
            for i in range(1, len(sentence.keys())):
                comp = parse(sentence[i], sentence, self.baseline)
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

    def get_enabled_feature(self, comp, idx_dict):
        """Returns enable feature id for (parent, child), if empty returns None"""
        key = self.extract_key(comp)
        if not key in self.feature_dict:
            return None
        return idx_dict[key]


class ParentWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (1, c['parent_word'], c['parent_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWord(FeatureFunction):

    def extract_key(self, c):
        key = (2, c['parent_word'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ParentPos(FeatureFunction):

    def extract_key(self, c):
        key = (3, c['parent_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (4, c['child_word'], c['child_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (5, c['child_word'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (6, c['child_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key

# bigram


class ParentChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (7, c['parent_word'], c['parent_pos'], c['child_word'], c['child_pos'], c['distance'])
        return key


class ParentPosChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (8, c['parent_pos'], c['child_word'], c['child_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWordChildWordPos(FeatureFunction):

    def extract_key(self, c):
        key = (9, c['parent_word'], c['child_word'], c['child_pos'], c['distance'])
        return key


class ParentWordPosChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (10, c['parent_word'], c['parent_pos'], c['child_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


class ParentWordPosChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (11, c['parent_word'], c['parent_pos'], c['child_word'], c['distance'])
        return key


class ParentChildWord(FeatureFunction):

    def extract_key(self, c):
        key = (12, c['parent_word'], c['child_word'], c['distance'])
        return key


class ParentChildPos(FeatureFunction):

    def extract_key(self, c):
        key = (13, c['parent_pos'], c['child_pos'], c['distance'])
        if self.baseline:
            return key[:-1]
        return key


# optional (could be None)
class PosNeighA(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (14, c['parent_pos'], c['n_parent_pos'], c['p_child_pos'], c['child_pos'], c['distance'])
        return key


class PosNeighB(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (15, c['p_parent_pos'], c['parent_pos'], c['p_child_pos'], c['child_pos'], c['distance'])
        return key


class PosNeighC(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (16, c['parent_pos'], c['n_parent_pos'], c['child_pos'], c['n_child_pos'], c['distance'])
        return key


class PosNeighD(FeatureFunction):

    @optional
    def extract_key(self, c):
        key = (17, c['p_parent_pos'], c['parent_pos'], c['child_pos'], c['n_child_pos'], c['distance'])
        return key


feature_fncs = [
    # unigram
    ('parent_word_pos', ParentWordPos),
    ('parent_word', ParentWord),
    ('parent_pos', ParentPos),
    ('child_word_pos',ChildWordPos),
    ('child_word', ChildWord),
    ('child_pos', ChildPos),
    # bigram
    ('parent_child_word_pos', ParentChildWordPos),
    ('parent_pos_child_word_pos', ParentPosChildWordPos),
    ('parent_word_child_word_pos', ParentWordChildWordPos),
    ('parent_word_pos_child_pos', ParentWordPosChildPos),
    ('parent_word_pos_child_word', ParentWordPosChildWord),
    ('parent_child_word', ParentChildWord),
    ('parent_child_pos', ParentChildPos),
    # extra
    ('pos_next_parent_previous_child', PosNeighA),
    ('pos_previous_parent_previous_child', PosNeighB),
    ('pos_next_parent_next_child', PosNeighC),
    ('pos_previous_parent_next_child', PosNeighD)
]


def init_feature_functions(train_data, filter_dict, baseline):
    callables_tuple_list = []
    feature_counts = {}
    i = 0
    for name, feature_func in feature_fncs:
        if name in filter_dict:
            callables_tuple_list.append((name, feature_func(name, train_data, baseline)))
            callables_tuple_list[i][1].filter_features(filter_dict[name])
            i += 1

    callables_dict = OrderedDict(callables_tuple_list)
    idx_dic = {}
    tmp_max = 0
    # build feature mapping
    for name, c in callables_dict.items():
        feature_counts = {**callables_dict[name].feature_dict, **feature_counts}
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


def get_features(sentence, child, parent_id, callables_dict, idx_dict, baseline):
    feature_indices = []
    tup = [child[0], child[1], child[2], parent_id]
    comp = parse(tup, sentence, baseline)
    for name, feature_function in callables_dict.items():
        feature_id = feature_function.get_enabled_feature(comp, idx_dict)
        if feature_id:
            feature_indices.append(feature_id)

    return feature_indices


def debug_features(idx_dict, w, feature_counts, model_name):
    print("Debugging features")
    s = ""
    for i, elem in enumerate(w):
        if elem == 0:
            rev_idx_dict = dict((v, k) for k, v in idx_dict.items())
            s += "Feature " + str(rev_idx_dict[i]) + " , count: " + str(feature_counts[rev_idx_dict[i]]) + "\n"
    with open("debug_features-" + model_name + ".log", 'a+') as handle:
        handle.write(s)
    # for name, feature_functions in callables_dict.items():
    #     print("10% lowest features counts in " + name)
    #     last_idx = int(len(feature_functions.feature_dict.items()) / 10)
    #     items = sorted(feature_functions.feature_dict.items(), key=operator.itemgetter(1), reverse=False)[:last_idx]
    #     print(items)
