import pandas as pd
import ast
import re
import copy
from random import sample


def remove_same(s: str):
    """
    remove parallel paragraphs in one line that are completely the same
    :param s: string of parallels
    :return : list of different parallel paragraphs
    """
    ls_parallel = s.split('《')
    for i in range(len(ls_parallel)):
        ls_parallel[i] = re.sub(r'^.+?》', '', ls_parallel[i], flags=re.S)
    rm_same = list(set(ls_parallel[1:]))
    return rm_same


def remove_similar(ls: list):
    """
    remove parallel paragraphs in a group that are highly similar
    :param x: list of parallel paragraphs
    :return: list of parallel paragraphs without highly similar ones
    """
    sorted_ls = sorted(ls, key=lambda x: len(x), reverse=False)  # 升序
    sorted_ori = copy.deepcopy(sorted_ls)
    re_punc = re.compile(r'[“”‘’，。？！·《》「」、『』]', flags=re.S)
    unsim_ls = []
    unsim_ori = []
    for i in range(len(ls)):
        sorted_ls[i] = re_punc.sub('', sorted_ls[i])
    for i in range(len(ls)):
        flag = 0
        if i < len(ls) - 1:
            if sorted_ls[i] in sorted_ls[i + 1]:
                continue
        for s in unsim_ls:
            if len(set(s) - set(sorted_ls[i])) <= 1 and len(set(sorted_ls[i]) - set(s)) <= 1:
                flag = 1
                break
        if flag == 0:
            unsim_ls.append(sorted_ls[i])
            unsim_ori.append(sorted_ori[i])

    return unsim_ori


def merge_parallel(x):
    all_parallel = []
    for i in range(len(x)):
        ls = remove_same(x['平行段落'].iloc[i])
        all_parallel += ls
    valid_parallel = remove_similar(all_parallel)
    return valid_parallel


if __name__ == "main":
    sheet_name = ['hanfei', 'shangshu', 'liji', 'zhaungzi', 'lunyu', 'mengzi+xunzi', 'zhanguoce', 'zuozhuan']

    # remove highly similar sentences in parallel groups
    for name in sheet_name:
        df_name = pd.read_excel('data/ctext平行段落.xlsx', sheet_name=name)
        df_merged = df_name.groupby(df_name['原句']).apply(merge_parallel)
        df_clean = df_merged.reset_index()
        df_clean.columns = ['原句', '平行段落']
        for i in range(len(df_clean) - 1, -1, -1):
            if len(df_clean['平行段落'].iloc[i]) < 2:
                df_clean.drop(index=i, inplace=True)
        writer = pd.ExcelWriter(r'data/ctext平行段落clean.xlsx', mode="a", engine="openpyxl")
        df_clean.to_excel(writer, index=False, sheet_name=name)
        writer.save()
        writer.close()

    # convert parallel groups to parallel pairs
    ls_pair = []
    cnt = 0
    for name in sheet_name:
        df = pd.read_excel('data/ctext平行段落clean.xlsx', sheet_name=name)
        N = len(df)
        for i in range(N):
            try:
                tmp_lst = ast.literal_eval(df['平行段落'].iloc[i])
                l = len(tmp_lst)
                print(l)
                if l < 2:
                    continue
                tmp_ls_pair = []
                # save at most 30 pairs
                for j in range(l):
                    for k in range(j + 1, l):
                        tmp_ls_pair.append([cnt, tmp_lst[j], tmp_lst[k]])
                if l > 8:
                    idx = sample(range(l * (l + 1) / 2), 30)
                    tmp_ls_pair = [tmp_ls_pair[i] for i in idx]
                ls_pair += tmp_ls_pair
            except:
                print(i, name, df['平行段落'].iloc[i])
            cnt += 1

    df_pair = pd.DataFrame(ls_pair, columns=['group_index', 'sentence1', 'sentence2'])
    print(f"get {len(df_pair)} pairs of parallels in total")
    df_pair.to_csv('data/ctext_parallel_pair.csv')
