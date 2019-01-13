from features import ParentWordPos, get_edge_features, init_feature_functions
import chu_liu
import numpy as np


class Perceptron:
    def __init__(self, sentences, x, y, ground_graphs, data, filter_dict, num_iter=10):
        self.num_iter = num_iter
        self.w = np.zeros(len(x[0]))
        self.sentences = sentences
        self.x = x
        self.y = y
        self.features_idx = []
        self.ground_graphs = ground_graphs
        self.callables_dict, self.idx_dict = init_feature_functions(data, filter_dict)

    def sentence_to_graph(self, sentence):
        graph = {0: {}}
        for key, item in enumerate(sentence):
            if key != 0:
                graph[key] = {}
                graph[0][key] = get_edge_features(sentence[key], item[0], sentence,
                                                  self.callables_dict, self.idx_dict)

        for child in sentence:
            for parent in sentence:
                if child == parent or child == 0 or parent == 0:
                    continue
                graph[parent][child] = get_edge_features(sentence[child], sentence[parent],
                                                         sentence, self.callables_dict, self.idx_dict)

        return graph

    def graph_to_features(self, graph, sentence):
        features = []
        for vertex, edges in graph.items():
            for neigh in edges:
                features += get_edge_features(sentence[neigh], sentence[vertex], sentence,
                                              self.callables_dict, self.idx_dict)

        return features

    def sentence_to_features(self, sentence):
        features = []
        sentence[0] = 0
        for idx, word in sentence.items():
            if idx == 0:
                continue
            features += get_edge_features(word, sentence[word.parent], sentence, self.callables_dict, self.idx_dict)

        return features

    def get_weighted_graph(self, graph):
        weighted_graph = {}
        for vertex, neigh in graph.items():
            weighted_graph[vertex] = {child: -np.sum(self.w[indices]) for child, indices in neigh.items()}

        return weighted_graph

    def fit(self):
        graphs = []

        for sentence in self.sentences:
            graphs.append([sentence, self.sentence_to_features(sentence), self.sentence_to_graph(sentence)])

        for i in range(self.num_iter):
            for idx, sentence, features, graph in enumerate(graphs):
                ground_graph = self.ground_graphs[idx]
                weighted_graph = self.get_weighted_graph(graph)
                w_graph = chu_liu.Digraph(weighted_graph)
                graph_mst = w_graph.mst()
                # Update part here
                if not self.compare_trees(graph_mst, ground_graph):
                    graph_features = self.graph_to_features(graph_mst, sentence)
                    for feature in features:
                        self.w[feature] += 1
                    for feature in graph_features:
                        self.w[feature] -= 1

        print('fit finished')

    @staticmethod
    def compare_trees(y_pred, y_true):
        for key, value in y_pred:
            for idx, item in enumerate(value):
                if item != y_true[key][idx]:
                    return False
        return True
