import torch
from sentence_transformers import util, SentenceTransformer, models
import csv

def evaluate_embedding(model, a, b):
    """
    evaluate the embedding quality of model.
    cos_sim(a[i], b[j]) is the cosine similarity matrix for a[i] and b[j],
    return two metrics:
    1) the gap between corresponding pairs and non-corresponding pairs, i.e.
        mean of cos_sim[i, i]-mean of cos_sim[i, j](i != j)
    2) the gap between cos_sim and identity matrix, i.e.
        mean of abs(identity_matrix[i, j] - cos_sim[i, j]) for all i, j
    :param a, b: list of sentences
    """
    assert len(a) == len(b)
    print("encoding sentences a......")
    a_embedding = model.encode(a)
    print("encoding sentences b......")
    b_embedding = model.encode(b)
    cossim = util.cos_sim(a_embedding, b_embedding)
    N = cossim.size()[0]
    A1 = torch.sum(torch.eye(N) * cossim) / N
    A2 = torch.sum((torch.ones(N, N) - torch.eye(N)) * cossim) / (N * (N - 1))
    A3 = torch.mean(torch.abs(torch.eye(N) - cossim))

    return A1 - A2, A3


def trad_simple_example(model):
    """
    An example to show the ability of the model to align sentences in
    traditional and simplified characters. The distance between sentences[1]
    and sentences[2] should be smaller than other pairs.
    :param model: model to test
    :return: matrix of cosine similarity
    """
    sentences = ['曾子曰：“慎終，追遠，民德歸厚矣。”',
                 '信近于义，言可复也。恭近于礼，远耻辱也。因不失其亲，亦可宗也。”',
                 '信近於義，言可復也。恭近於禮，遠恥辱也。因不失其親，亦可宗也。']
    embeddings = model.encode(sentences)
    cos_sim = util.cos_sim(embeddings, embeddings)
    return cos_sim


def eval_model(model):
    trad, simple = [], []
    with open("data/char_test.csv", "rt", encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trad.append(row['traditional'])
            simple.append(row['simplified'])

    src_ls, tgt_ls = [], []
    with open("data/cam_dev_triple.csv", "rt", encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            src_ls.append(row['src_sentence'])
            tgt_ls.append(row['tgt_sentence'])

    ts_metric1, ts_metric2 = evaluate_embedding(model, trad, simple)
    print(f"On traditional-simplified Chinese character parallels, metric1 = {ts_metric1}, metric2 = {ts_metric2}")
    cam_metric1, cam_metric2 = evaluate_embedding(model, src_ls, tgt_ls)
    print(f"On Chinese ancient-modern parallel corpus, cam_metric1 is: {cam_metric1}, cam_metric2 is: {cam_metric2}")
    cossim = trad_simple_example(model)
    print(f"traditional-simplified Chinese character parallel example:")
    print(cossim)


device = "cuda" if torch.cuda.is_available() else "cpu"

# It is just an example here.
# You may want to modify the model names and the PATH.

for model_name in ['model_after_ctext', 'model_after_traditional_simple_parallel', 'model_after_cam_parallel']:
    PATH = './finetuned_model/' + model_name
    print(f"loading model {model_name}...")
    model = SentenceTransformer(PATH).to(device)
    print(f"Testing model {model_name}......")
    eval_model(model)