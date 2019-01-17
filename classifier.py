from features import ParentWordPos, get_edge_features, init_feature_functions, compute_features_size
import chu_liu
import numpy as np
from collections import defaultdict
from common import pickle_save, pickle_load, timeit

class Perceptron:

    @timeit
    def __init__(self, sentences, filter_dict):
        self.num_iter = 1
        self.sentences = sentences
        self.features_idx = []
        self.ground_graphs = {}
        self.get_ground_graphs(sentences)
        self.callables_dict, self.idx_dict = init_feature_functions(sentences, filter_dict)
        self.m = compute_features_size(self.callables_dict)
        self.w = np.zeros(self.m)
        # in-training measures
        self.true = 0
        self.false = 0
        self.total = 0
        self.accuracy = 0

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
                features += get_edge_features(sentence, sentence[neigh], sentence[vertex][0],
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
            weighted_graph[vertex] = {child: np.sum(self.w[indices]) for child, indices in neigh.items()}

        return weighted_graph

    def get_ground_graphs(self, data):
        for sentence_idx, sentence in enumerate(data):
            for word_idx, word in sentence.items():
                if word_idx == 0:
                    self.ground_graphs[sentence_idx] = {}
                    self.ground_graphs[sentence_idx][0] = []
                if word[3] not in self.ground_graphs[sentence_idx].keys():
                    self.ground_graphs[sentence_idx][word[3]] = []
                if word[0] not in self.ground_graphs[sentence_idx].keys():
                    self.ground_graphs[sentence_idx][word[0]] = []
                if word_idx != 0:
                    self.ground_graphs[sentence_idx][word[3]].append(word[0])

    @timeit
    def fit(self, num_iter=10):
        self.true = 0
        self.false = 0
        self.total = 0
        self.accuracy = 0
        self.num_iter = num_iter
        graphs = []

        for sentence in self.sentences:
            graphs.append({'sentence': sentence, 'sent_feat': self.sentence_to_features(sentence),
                           'sent_graph': self.sentence_to_graph(sentence)})

        for i in range(self.num_iter):
            for idx, graph_dict in enumerate(graphs):
                ground_graph = self.ground_graphs[idx]
                weighted_graph = self.get_weighted_graph(graph_dict['sent_graph'])

                full_graph = {}

                for parent_id, parent in graph_dict['sent_graph'].items():
                    for child_id in parent.keys():
                        if parent_id not in full_graph:
                            full_graph[parent_id] = []
                        if child_id == 0:
                            continue
                        full_graph[parent_id].append(child_id)

                def get_score(h, m):
                    return weighted_graph[h][m]

                w_graph = chu_liu.Digraph(full_graph, get_score=get_score)
                graph_mst = w_graph.mst()
                # Update part here
                if not self.compare_trees(graph_mst, ground_graph):
                    graph_features = self.graph_to_features(graph_mst, graph_dict['sentence'])
                    for feature in graph_dict['sent_feat']:
                        self.w[feature] += 1
                    for feature in graph_features:
                        self.w[feature] -= 1
                self.update_accuracy(graph_mst, ground_graph)

            iter = i + 1
            self.print_stats(iter)

            if iter in [20, 50, 80, 100]:
                pickle_save(self.w, 'w%d.pickle' % iter)

        print('fit finished')
        print(self.w)

    @staticmethod
    def compare_trees(y_pred, y_true):
        flg = False
        for key, value in y_pred.successors.items():
            if value != y_true[key]:
                return False
        for key, value in y_pred.successors.items():
            if value:
                flg = True

        return flg

    def update_accuracy(self, y_pred, y_true):
        y_pred = y_pred.successors
        for key, value in y_true.items():
            if value:
                for item in value:
                    if item in y_pred[key]:
                        self.true += 1
                    else:
                        self.false += 1
                    self.total += 1

        self.accuracy = self.true/self.total

    def print_stats(self, iter):
        print("finished iter " + str(iter))
        print("Current accuracy: %f" % self.accuracy)
        w_status = (np.sum(self.w > 0), np.sum(self.w < 0), np.sum(self.w == 0))
        print("Weights status: Pos=%d, Neg=%d, Zero=%d" % w_status)
