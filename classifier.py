from features import get_features, init_feature_functions, compute_features_size
import chu_liu
import numpy as np
from common import pickle_save, timeit

EARLY_STOPPING_ITERATIONS = 20


class Perceptron:
    def __init__(self, train_data, test_data, filter_dict, baseline, early_stopping):
        self.num_iter = 1
        self.train_data = train_data
        self.test_data = test_data
        self.features_idx = []
        self.ground_graphs = {}
        self.get_ground_graphs(train_data)
        self.baseline = baseline
        self.callables_dict, self.idx_dict = init_feature_functions(train_data, filter_dict, baseline)
        self.m = compute_features_size(self.callables_dict)
        self.w = np.zeros(self.m)
        # in-training measures
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.best_accuracy = 0
        self.iter_num = 0
        self.iter_no_change = 0
        self.early_stopping = early_stopping

    @timeit
    def fit(self, num_iter=10):
        self.num_iter = num_iter
        graphs = []

        for sentence in self.train_data:
            graphs.append({'sentence': sentence, 'sent_feat': self.sentence_to_features(sentence),
                           'sent_graph': self.sentence_to_graph(sentence)})

        for i in range(1, self.num_iter+1):
            for idx, graph_dict in enumerate(graphs):
                ground_graph = self.ground_graphs[idx]
                weighted_graph = self.get_weighted_graph(graph_dict['sent_graph'])
                full_graph = self.get_full_graph(graph_dict)

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

            self.update_measures()
            self.print_stats(i)
            # keep w
            if self.test_accuracy > self.best_accuracy:
                print("Reached best accuracy, saving model")
                self.best_accuracy = self.test_accuracy
                pickle_save(self.w, 'w.pickle')
            else:
                self.iter_no_change += 1

            if self.iter_no_change > EARLY_STOPPING_ITERATIONS:
                print("Breaking training loop due to early stopping")
                break

            if i in [20, 50, 80, 100]:
                if self.baseline:
                    pickle_save(self.w, 'w%d.pickle' % i)

        print("Fit finished, best test accuracy: %f" % self.best_accuracy)

    def predict(self, data):
        graphs = []
        graphs_mst = []

        for sentence in data:
            graphs.append({'sent_graph': self.sentence_to_graph(sentence)})

        for idx, graph_dict in enumerate(graphs):
            weighted_graph = self.get_weighted_graph(graph_dict['sent_graph'])
            full_graph = self.get_full_graph(graph_dict)

            def get_score(h, m):
                return weighted_graph[h][m]

            w_graph = chu_liu.Digraph(full_graph, get_score=get_score)
            graphs_mst.append(w_graph.mst())

        return graphs_mst

    def sentence_to_graph(self, sentence):
        graph = {0: {}}
        for key, item in sentence.items():
            if key != 0:
                graph[key] = {}
                graph[0][key] = get_features(sentence, item, item[3],
                                             self.callables_dict, self.idx_dict)

        for child in sentence.values():
            for parent in sentence.values():
                if child[0] == parent[0] or child[0] == -1 or parent[0] == -1:
                    continue
                if parent[0] not in graph:
                    graph[parent[0]] = {}
                graph[parent[0]][child[0]] = get_features(sentence, child, parent[0],
                                                          self.callables_dict, self.idx_dict)

        return graph

    def graph_to_features(self, graph, sentence):
        features = []
        for vertex, edges in graph.successors.items():
            for neigh in edges:
                features += get_features(sentence, sentence[neigh], sentence[vertex][0],
                                         self.callables_dict, self.idx_dict)

        return features

    def sentence_to_features(self, sentence):
        features = []
        for idx, word in sentence.items():
            if idx == 0:
                continue
            features += get_features(sentence, word, word[3], self.callables_dict, self.idx_dict)

        return features

    def get_weighted_graph(self, graph):
        weighted_graph = {}
        for vertex, neigh in graph.items():
            weighted_graph[vertex] = {child: np.sum(self.w[indices]) for child, indices in neigh.items()}

        return weighted_graph

    def get_full_graph(self, graph_dict):
        full_graph = {}
        for parent_id, parent in graph_dict['sent_graph'].items():
            for child_id in parent.keys():
                if parent_id not in full_graph:
                    full_graph[parent_id] = []
                if child_id == 0:
                    continue
                full_graph[parent_id].append(child_id)

        return full_graph

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

    @staticmethod
    def ground_graph_test(data):
        ground_graphs = {}
        for sentence_idx, sentence in enumerate(data):
            for word_idx, word in sentence.items():
                if word_idx == 0:
                    ground_graphs[sentence_idx] = {}
                    ground_graphs[sentence_idx][0] = []
                if word[3] not in ground_graphs[sentence_idx].keys():
                    ground_graphs[sentence_idx][word[3]] = []
                if word[0] not in ground_graphs[sentence_idx].keys():
                    ground_graphs[sentence_idx][word[0]] = []
                if word_idx != 0:
                    ground_graphs[sentence_idx][word[3]].append(word[0])

        return ground_graphs

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

    def update_measures(self):
        train_y_true = self.ground_graphs
        train_y_pred = self.predict(self.train_data)
        self.train_accuracy = self.get_accuracy(train_y_pred, train_y_true)

        test_y_true = self.ground_graph_test(self.test_data)
        test_y_pred = self.predict(self.test_data)
        self.test_accuracy = self.get_accuracy(test_y_pred, test_y_true)

    def print_stats(self, iter_num):
        print("finished iter " + str(iter_num))
        print("Current accuracy: %f" % self.train_accuracy)
        print("Test accuracy: %f" % self.test_accuracy)
        w_status = (np.sum(self.w > 0), np.sum(self.w < 0), np.sum(self.w == 0))
        print("Weights status: Pos=%d, Neg=%d, Zero=%d" % w_status)

    def get_accuracy(self, y_pred, y_true):
        true = 0
        total = 0

        for idx, pred in enumerate(y_pred):
            y_pred_succ = pred.successors

            for key, value in y_true[idx].items():
                if value:
                    for item in value:
                        if item in y_pred_succ[key]:
                            true += 1
                        total += 1

        return true / total
