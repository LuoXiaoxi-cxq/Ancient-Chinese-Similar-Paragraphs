import pandas as pd
from sentence_transformers import util, SentenceTransformer
import csv
import time
import torch
from function import empty_cache


def make_corpus(max_size=-1):
    """
    make the content for clustering
    :param max_size: the max size of sentences to cluster, default none
    :return:
    corpus_sentences: list, containing the '内容' field of zuozhuan, guoyu,
    zhanguoce csv files
    corpus_dict: dict, idx -> list(idx of paragraph, idx of sentence, sentence)
    """
    corpus_sentences = []
    dataset_name = ['zuozhuan', 'guoyu', 'zhanguoce']
    corpus_dict = dict()

    # csv file has three fields: '段落', '小句', '内容'
    cnt = 0
    for name in dataset_name:
        with open(f'data/{name}_sentence.csv', encoding='utf8') as f:
            reader = csv.DictReader(f, quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                corpus_sentences.append(row['内容'])
                corpus_dict[cnt] = [row['段落'], row['小句'], row['内容']]
                cnt += 1
                if 0 < max_size < cnt:
                    print("reach max_size, stop")
                    return corpus_sentences, corpus_dict
    return corpus_sentences, corpus_dict


def cluster(model, model_name, batchsize=64, max_size=-1):
    """
    Cluster with a specific model. Save the clustering result under './result/clustering/'.
    :param model: the model used for clustering
    """
    corpus, corpus_dict = make_corpus(max_size)
    corpus_embeddings = model.encode(corpus, batch_size=batchsize, show_progress_bar=True, convert_to_tensor=True)
    print(f"Start clustering, {len(corpus)} sentences in total")
    start_time = time.time()

    # 'threshold' Parameters to tune:
    # Only consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.8)

    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    # save the result of clustering as file
    ls_cluster = []
    print(f"Get {len(clusters)} clusters in total.")
    for i, cluster in enumerate(clusters):
        print("Cluster {}, #{} Elements ".format(i, len(cluster)))
        for idx in cluster:
            ls_cluster.append([i] + corpus_dict[idx])
        ls_cluster.append(['', '', '', ''])  # add an empty line
    df_cluster = pd.DataFrame(ls_cluster, columns=['聚类序号', '段落', '小句', '内容'])
    df_cluster.to_excel('./result/clustering/' + model_name + '.xlsx', index=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
empty_cache()

# It is just an example here.
# You may want to modify the model names and the PATH.

for model_name in ['model_after_traditional_simple_parallel']:
    PATH = './finetuned_model/' + model_name
    print(f"loading model {model_name}...")
    model = SentenceTransformer(PATH).to(device)
    cluster(model, model_name)
    empty_cache()
