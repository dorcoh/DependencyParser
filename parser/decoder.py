from parser.common import get_text_file


class Dataset(object):
    """Class that iterates over Dataset
    __iter__ method yields a tuple (words, tags)
        lst: list of words/tags
        sentence: the text of the sentence
    If processing_word and processing_tag are not None,
    optional pre-processing is applied
    Example:
        ```python
        data = Dataset(filename)
        for tuples, sentence in data:
            pass
        ```
    """

    def __init__(self, filename, max_iter=None, comp=False, slice=None):
        """
        Args:
            filename: path to the file
            max_iter: (optional) max number of sentence to yield
            slice: expects indices range e.g., (0,1000) -> 0:1000, assumes correct sizes
        """
        self.filename = filename
        self.max_iter = max_iter
        self.length = None
        self.comp = comp
        self.slice = slice

    def __iter__(self):
        pass

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

    def __str__(self):
        st = ''
        for sentence, tag, pos in self:
            st += ' '.join(sentence) + '\n' + ' '.join(tag) + '\n' + ' '.join(pos) + '\n'

        return st


class Data(Dataset):
    def __iter__(self):
        txt = get_text_file(self.filename)
        sentence = {}
        for row in txt:
            if not row:
                yield sentence
                sentence = {}
            else:
                if not self.comp:
                    if row[0] == '1':
                        # sentence.append(['-1', 'ROOT', 'ROOT', '-2'])
                        sentence[0] = [0, 'ROOT', 'ROOT', 0]
                        sentence[int(row[0])] = [int(row[0]), row[1], row[3], int(row[6])]

                    else:
                        sentence[int(row[0])] = [int(row[0]), row[1], row[3], int(row[6])]
                else:
                    sentence[int(row[0])] = ([int(row[0]), row[1], row[3]])

    def get_ground_tree(self):
        tree = []

        for sentence in self:
            tmp = {}
            for word in sentence:
                tmp[word[0]] = [word[3]]
            tree.append(tmp)

        return tree
