import csv

# The original corpus is in this :
# https://github.com/dayihengliu/a2m_chineseNMT/tree/master

# train set
f_train_src = open('../chinese_ancient_modern/train_src.txt', 'r', encoding='UTF-8')
f_train_tgt = open('../chinese_ancient_modern/train_tgt.txt', 'r', encoding='UTF-8')
header = ['src_sentence', 'tgt_sentence']

with open('data/cam_train.csv', 'w', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    ls_src = f_train_src.readlines()
    ls_tgt = f_train_tgt.readlines()
    for i in range(3):
        ls_src[i] = ls_src[i].replace(' ', '').strip()
    for i in range(2, len(ls_src) - 1):
        ls_src[i + 1] = ls_src[i + 1].replace(' ', '').strip()
        if not ls_src[i] or len(ls_src[i]) <= 5:
            continue
        if (ls_src[i] in ls_src[i + 1]) or (ls_src[i] in ls_src[i - 1]) or (ls_src[i] in ls_src[i - 2]):
            continue
        else:
            l_tgt = ls_tgt[i].replace(' ', '').strip()
            writer.writerow([ls_src[i], l_tgt])
    f_train_src.close()
    f_train_tgt.close()

# append original test set to train set
f_test_src = open('../chinese_ancient_modern/test_src.txt', 'r', encoding='UTF-8')
f_test_tgt = open('../chinese_ancient_modern/test_tgt.txt', 'r', encoding='UTF-8')

with open('data/cam_train.csv', 'a+', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    ls_src = f_test_src.readlines()
    ls_tgt = f_test_tgt.readlines()
    for i in range(3):
        ls_src[i] = ls_src[i].replace(' ', '').strip()
    for i in range(2, len(ls_src) - 1):
        ls_src[i + 1] = ls_src[i + 1].replace(' ', '').strip()
        if not ls_src[i] or len(ls_src[i]) <= 5:
            continue
        if (ls_src[i] in ls_src[i + 1]) or (ls_src[i] in ls_src[i - 1]) or (ls_src[i] in ls_src[i - 2]):
            continue
        else:
            l_tgt = ls_tgt[i].replace(' ', '').strip()
            writer.writerow([ls_src[i], l_tgt])
    f_test_src.close()
    f_test_tgt.close()

# dev set
f_train_src = open('../chinese_ancient_modern/dev_src.txt', 'r', encoding='UTF-8')
f_train_tgt = open('../chinese_ancient_modern/dev_tgt.txt', 'r', encoding='UTF-8')
header = ['src_sentence', 'tgt_sentence']

cnt = 0
ls_src = f_train_src.readlines()
ls_tgt = f_train_tgt.readlines()
f_train_src.close()
f_train_tgt.close()

with open('data/cam_test.csv', 'w', encoding='UTF-8', newline='') as f1, open('data/cam_train.csv', 'a',
                                                                              encoding='UTF-8', newline='') as f2:
    writer_test = csv.writer(f1)
    writer_train = csv.writer(f2)
    # write the header
    writer_test.writerow(header)
    for i in range(3):
        ls_src[i] = ls_src[i].replace(' ', '').strip()
    for i in range(2, len(ls_src) - 1):
        ls_src[i + 1] = ls_src[i + 1].replace(' ', '').strip()
        if not ls_src[i] or len(ls_src[i]) <= 5:
            continue
        if (ls_src[i] in ls_src[i + 1]) or (ls_src[i] in ls_src[i - 1]) or (ls_src[i] in ls_src[i - 2]):
            continue
        else:
            l_tgt = ls_tgt[i].replace(' ', '').strip()
            if cnt < 2000:
                writer_test.writerow([ls_src[i], l_tgt])
            else:
                writer_train.writerow([ls_src[i], l_tgt])
            cnt += 1