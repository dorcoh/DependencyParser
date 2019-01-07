from abc import ABC, abstractmethod


class FeatureFunction(ABC):
    def __init__(self):
        self.feature_dict = {}
        self.name = None

    def compute_size(self):
        """should fill the dictionary before calling this"""
        return len(self.features_dict.keys())

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
        for tup in sentence:
            parentid = int(tup[3])
            parent_word = sentence[parentid][1]
            parent_pos = sentence[parentid][2]
            key = (parent_word, parent_pos)
            if key in self.feature_dict:
                self.feature_dict[key] += 1
            else:
                self.feature_dict[key] = 0
