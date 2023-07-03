import docx
import re
import random
import zhconv
import csv

doc = docx.Document("先秦古籍編年.docx")

corpus = []
for para in doc.paragraphs:
    if para.style.name in ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Heading 5', 'Heading 6']:
        continue
    else:
        txt = para.text.strip()
        if not txt:
            continue
        txt.replace("\n", "")
        txt = re.sub(r"[\d·.“”‘’ ]", "", txt)
        txt = re.sub(r"（.*?）|\(.*?\)|〔.*?〕|\[.*?\]", "", txt)
        prob = random.random()
        if len(txt) >= 30 or prob <= len(txt) / 50:
            sentences = re.split(r'？”|！”|……”|[。！？；]', txt)
            for i in range(len(sentences)-1,-1,-1):
                s_= re.sub(r"[，,《》“”‘’·「」、『』]", "", sentences[i])
                if not s_:
                    del sentences[i]
            if len(sentences) <= 4:
                sen = sentences
            else:
                idx = random.sample(range(len(sentences)), 4)
                sen = [sentences[i] for i in idx]
            sen = list(map(lambda x: x[:50] if len(x) > 50 else x, sen))
            corpus += sen

random.shuffle(corpus)

test_set = []
for i in range(2000):
    trad_s = corpus[i]
    test_set.append([trad_s, zhconv.convert(trad_s, 'zh-cn')])

with open(r'data/char_test.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    header = ['traditional', 'simplified']
    writer.writerow(header)
    for ls in test_set:
        writer.writerow(ls)

train_set = []
for i in range(2000, len(corpus)):
    trad_s = corpus[i]
    test_set.append([trad_s, zhconv.convert(trad_s, 'zh-cn')])

with open(r'data/char_train.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    header = ['traditional', 'simplified']
    writer.writerow(header)
    for ls in test_set:
        writer.writerow(ls)