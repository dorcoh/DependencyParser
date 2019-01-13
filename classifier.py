from features import ParentWordPos, get_edge_features, init_feature_functions
import chu_liu
import numpy as np


class Perceptron:
    def __init__(self, sentences, ground_graphs, filter_dict, num_iter=10):
        self.num_iter = num_iter
        self.sentences = sentences
        self.features_idx = []
        self.ground_graphs = ground_graphs
        self.callables_dict, self.idx_dict = init_feature_functions(sentences, filter_dict)
        #self.w = np.zeros(len(self.idx_dict.keys()))   # TODO: filtering occurrences causes missing indices in dict (OR we just delete those indices somewhere)
        self.w = np.zeros(max(self.idx_dict.values())+1)

    def sentence_to_graph(self, sentence):
        graph = {0: {}}
        for key, item in sentence.items():
            if key != 0:
                graph[key] = {}
                graph[0][key] = get_edge_features(sentence, item, item[3],
                                                  self.callables_dict, self.idx_dict)

        for child in sentence.values():
            for parent in sentence.values():
                if child[0] == parent[0] or child[0] == -1 or parent[0] == -1:
                    continue
                if parent[0] not in graph:
                    graph[parent[0]] = {}
                graph[parent[0]][child[0]] = get_edge_features(sentence, child, parent[0],
                                                         self.callables_dict, self.idx_dict)

        return graph

    def graph_to_features(self, graph, sentence):
        features = []
        for vertex, edges in graph.successors.items():
            for neigh in edges:
                features += get_edge_features(sentence, sentence[neigh], sentence[vertex],
                                              self.callables_dict, self.idx_dict)

        return features

    def sentence_to_features(self, sentence):
        features = []
        for idx, word in sentence.items():
            if idx == 0:
                continue
            features += get_edge_features(sentence, word, word[3], self.callables_dict, self.idx_dict)

        return features

    def get_weighted_graph(self, graph):
        weighted_graph = {}
        for vertex, neigh in graph.items():
            weighted_graph[vertex] = {child: -np.sum(self.w[indices]) for child, indices in neigh.items()}

        return weighted_graph

    def fit(self):
        graphs = []

        for sentence in self.sentences:
            graphs.append({'sentence': sentence, 'sent_feat': self.sentence_to_features(sentence),
                           'sent_graph': self.sentence_to_graph(sentence)})

        for i in range(self.num_iter):
            for idx, graph_dict in enumerate(graphs):
                ground_graph = self.ground_graphs[idx]
                weighted_graph = self.get_weighted_graph(graph_dict['sent_graph'])
                w_graph = chu_liu.Digraph(weighted_graph)
                graph_mst = w_graph.mst()
                # Update part here
                if not self.compare_trees(graph_mst, ground_graph):
                    graph_features = self.graph_to_features(graph_mst, graph_dict['sentence'])
                    for feature in graph_dict['sent_feat']:
                        self.w[feature] += 1
                    for feature in graph_features:
                        self.w[feature] -= 1
        print('fit finished')
        print(self.w)

    @staticmethod
    def compare_trees(y_pred, y_true):
        flg = False
        for key, value in y_pred.successors.items():
            for idx, item in enumerate(value):
                if item != y_true[key][idx]:
                    return False
        for key, value in y_pred.successors.items():
            if value:
                flg = True
        return flg
