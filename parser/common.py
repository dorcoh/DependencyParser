import csv
import pickle
import time


def get_text_file(file_path):
    rows = []

    with open(file_path, 'r') as data_tsv:
        d_tsv = csv.reader(data_tsv, delimiter='\t')
        for row in d_tsv:
            rows.append(row)

    return rows


def pickle_load(filename):
    try:
        with open(filename, 'rb') as handle:
            pickled = pickle.load(handle)
            return pickled
    except Exception as e:
        print("Pickle load failed with filename: " + str(filename))
        print("Exception raised: " + str(e))
        return None


def pickle_save(obj, filename):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
    except Exception as e:
        print("Pickle save failed with filename, object: " + str(filename) + ',' + str(obj))
        print("Exception raised: " + str(e))
        return None


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def transpose_tree(tree):
    tmp_dic = {}

    for key, value in tree.items():
        for item in value:
            tmp_dic[item] = key

    return tmp_dic


def tag_comp(y_pred, test_sent, file_name):
    with open(file_name, mode='w', newline='') as comp_file:
        writer = csv.writer(comp_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for sentence_idx, sentence in enumerate(test_sent):
            successors = y_pred[sentence_idx].successors
            transposed_tree = transpose_tree(successors)
            for word_idx, word in sentence.items():
                if word_idx == 0:
                    continue
                writer.writerow([word_idx, word[1], '_', word[2], '_', '_', transposed_tree[word_idx], '_', '', '_'])
            writer.writerow([])
