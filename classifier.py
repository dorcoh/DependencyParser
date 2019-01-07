from features import ParentWordPos

class Perceptron:
    def __init__(self, num_iter=10):
        self.num_iter = num_iter
        self.w = 0

    def fit(self, x, y):
        k = 0

        for i in range(self.num_iter):
            for j in range(len(y)):
                pass
                # Update part here
                if not self.compare_trees(y_pred, y[j]):
                    self.w = None
                    k += 1

        print('fit finished')

    @staticmethod
    def compare_trees(y_pred, y_true):
        for key, value in y_pred:
            for idx, item in enumerate(value):
                if item != y_true[key][idx]:
                    return False
        return True
