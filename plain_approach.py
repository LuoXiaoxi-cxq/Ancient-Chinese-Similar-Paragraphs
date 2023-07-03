import pandas as pd
from transformers import pipeline
import ast
import math
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


classifier = pipeline("token-classification", model="ckiplab/bert-base-han-chinese-ws-shanggu")



csv_raw=pd.read_csv('data/zhanguoce_sentence.csv')
zhanguoce_split=csv_raw.values.tolist()


split_processed=[]
for i in range(len(zhanguoce_split)):
    #print(i)
    line=zhanguoce_split[i]
    sent=line[2]
    sent_ws=classifier(sent)
    word=""
    wordsplit=[]
    for char in sent_ws:
        if char['entity']=='B':
            if word!="":
                wordsplit.append(word)
            word=""
        word=word+char['word']
    if word!="":
        wordsplit.append(word)
    line2=[i]+line
    line2.append(wordsplit)
    split_processed.append(line2)

#df_split_p = pd.DataFrame(split_processed, columns=['编号','段落', '小句','内容','切分'])
#df_split_p.to_csv('战国策_小句2.csv',index=False,encoding='UTF-8')
zhanguoce_split=split_processed
for i in zhanguoce_split:
    i[1]="战"+str(i[1])



csv_raw=pd.read_csv('data/guoyu_sentence.csv')
guoyu_split=csv_raw.values.tolist()


split_processed=[]
for i in range(len(guoyu_split)):
    #print(i)
    line=guoyu_split[i]
    sent=line[2]
    sent_ws=classifier(sent)
    word=""
    wordsplit=[]
    for char in sent_ws:
        if char['entity']=='B':
            if word!="":
                wordsplit.append(word)
            word=""
        word=word+char['word']
    if word!="":
        wordsplit.append(word)
    line2=[i]+line
    line2.append(wordsplit)
    split_processed.append(line2)

#df_split_p = pd.DataFrame(split_processed, columns=['编号','段落', '小句','内容','切分'])
#df_split_p.to_csv('国语_小句2.csv',index=False,encoding='UTF-8')

guoyu_split=split_processed
for i in guoyu_split:
    i[1]="国"+str(i[1])




csv_raw=pd.read_csv('data/zuozhuan_sentence.csv')
zuozhuan_split=csv_raw.values.tolist()

split_processed=[]
for i in range(len(zuozhuan_split)):
    #print(i)
    line=zuozhuan_split[i]
    sent=line[2]
    sent_ws=classifier(sent)
    word=""
    wordsplit=[]
    for char in sent_ws:
        if char['entity']=='B':
            if word!="":
                wordsplit.append(word)
            word=""
        word=word+char['word']
    if word!="":
        wordsplit.append(word)
    line2=[i]+line
    line2.append(wordsplit)
    split_processed.append(line2)

#df_split_p = pd.DataFrame(split_processed, columns=['编号','段落', '小句','内容','切分'])
#df_split_p.to_csv('左传_小句2.csv',index=False,encoding='UTF-8')

zuozhuan_split=split_processed
for i in zuozhuan_split:
    i[1]="左"+str(i[1])



split_processed=zuozhuan_split+zhanguoce_split+guoyu_split
for i in range(len(split_processed)):
    split_processed[i][0]=i

#df_split_p = pd.DataFrame(split_processed, columns=['编号','段落', '小句','内容','切分'])
#df_split_p.to_csv('左传+战国策+国语_小句2.csv',index=False,encoding='UTF-8')




grams={}
stop={",","："}
for i in split_processed:
    wordtable=i[4]
    for word in wordtable:
        if word in stop:
            continue
        if word in grams:
            grams[word]=grams[word]+1
        else:
            grams[word]=1


tf_idf=[]
for i in split_processed:
    vec=[]
    for word in grams:
        if word in i[4]:
            tf=0.0
            tf=i[4].count(word)/len(i[4])
            idf=0.0
            idf=math.log(len(i[4])/(grams[word]))
            vec.append(tf*idf)
        else:
            vec.append(0)
    tf_idf.append(vec)



clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.75) #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(tf_idf)
cluster_assignment = clustering_model.labels_


def calculate(i,j):
    sent1=split_processed[i][4]
    sent2=split_processed[j][4]

    if len(sent1)<4 or len(sent2)<4:
        return

    score=0.0

    vec1=np.array(tf_idf[i])
    vec2=np.array(tf_idf[j])
    score2 = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    if score2<=0.25:
        return

    #bigram
    bigram1=set()
    bigram2=set()
    for k in range(len(sent1)-1):
        if sent1[k] not in stop and sent1[k+1] not in stop:
            bigram1.add(sent1[k]+sent1[k+1])
    for k in range(len(sent2)-1):
        if sent2[k] not in stop and sent2[k+1] not in stop:
            bigram2.add(sent2[k]+sent2[k+1])
    bigram_union=bigram1 | bigram2
    bigram_inter=bigram1 & bigram2
    bigram_rat=0.0
    if len(bigram_union)!=0:
        bigram_rat=len(bigram_inter)/len(bigram_union)

    #trigram
    trigram1=set()
    trigram2=set()
    for k in range(len(sent1)-2):
        if sent1[k] not in stop and sent1[k+1] not in stop and sent1[k+2] not in stop: 
            trigram1.add(sent1[k]+sent1[k+1]+sent1[k+2])
    for k in range(len(sent2)-2):
        if sent2[k] not in stop and sent2[k+1] not in stop and sent2[k+2] not in stop:
            trigram2.add(sent2[k]+sent2[k+1]+sent2[k+2])
    trigram_union=trigram1 | trigram2
    trigram_inter=trigram1 & trigram2
    trigram_rat=0.0
    if len(trigram_union)!=0:
        trigram_rat=len(trigram_inter)/len(trigram_union)
    
    #具体的score怎么计算and阈值怎么设定？
    score1=bigram_rat*0.4+trigram_rat*0.6


    score=0.5*(score1+score2)
    if score>0.7:
        paral_list.append(split_processed[i][:4]+split_processed[j][:4])



clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentence_id)

paral_list=[]
for si in range(len(clustered_sentences)):
    print(si)
    s=clustered_sentences[si]
    if len(s)<=1:
        continue
    for i in range(len(s)-1):
        for j in range(i+1,len(s)):
            calculate(s[i],s[j])

paral_list_df = pd.DataFrame(paral_list, columns=['编号1','段落序号1', '小句序号1','内容1','编号2','段落序号2', '小句序号2','内容2'])
paral_list_df.to_csv('data/zuozhuan+zhanguoce+guoyu_process.csv',index=False,encoding='UTF-8')