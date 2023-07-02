import torch
import gc
import csv
from sentence_transformers import InputExample


class MySampler(torch.utils.data.Sampler):
    """
    Define sampler for ctext training set.
    Sample elements in sequence of:
     0, 30, 60, 90,...
     1, 31, 61, 91,...
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        l = len(self.data_source)
        idx_ls = []
        for i in range(30):
            for j in range(l // 30 + 1):
                if j * 30 + i < l:
                    idx_ls.append(j * 30 + i)

        return iter(idx_ls)

    def __len__(self):
        return len(self.data_source)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def make_training_set(s):
    """
    make training set from csv files
    :param s: name of csv file under ./data/ directory
    :return: training set
    """
    assert s in ['ctext_parallel_pair', 'char_train', 'cam_train']
    training_set = []
    csv_reader = csv.reader(open("./data/" + s + ".csv", encoding='UTF-8'))
    for line in csv_reader:
        if s == 'ctext_parallel_pair':
            line = line[2:]
        training_set.append(InputExample(texts=line))
        if s != 'ctext_parallel_pair':
            del training_set[0]
    return training_set
