# -*- coding:utf-8 -*-

import re
import pandas as pd
import requests
import time

range_dict = {'analects': range(1102, 1601), 'mengzi': range(1604, 1878), 'zuozhuan': range(16954, 20906),
              'shangshu': range(21032, 21506), 'xunzi': range(12247, 12847), 'hanfeizi': range(1881, 2701),
              'liji': range(9481, 10471), 'zhanguoce': range(49466 - 50757), 'guoyu': range(24452, 24982),
              'zhaungzi': range(2717, 3012), 'zhuangzi2': range(42596, 42619)}

range_dict = {'liji1': range(9481, 9981)}


# analects 1102-1601
# mengzi 1604-1878
# xunzi 12247-12847
# hanfeizi 1881-2701
# guoyu 24452-24982
# zuozhaun 16954-20906
# shangshu 21032-21505
# liji 9481-10471
# zhaungzi  2717-3012 42596-42619
# zhanguoce  49466-50757

def get_content(range_name, range_list):
    cnt = 0
    cnt2 = 0
    parallel = []
    with open('data/log.txt', 'a', encoding='UTF-8') as logfile:
        for i in range_list:
            cnt += 1
            url = 'https://ctext.org/text.pl?node=' + str(i) + '&if=en&show=parallel'
            headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                'Referer': "https://ctext.org/"}
            try:
                response = requests.get(url, headers=headers, verify=False)
            except Exception as e:
                print(f"catch an exception when i = {i} ", e)
                continue

            response_txt = response.text
            re1 = re.compile(r'onclick="updateGraph\(\d+\);">(.*)<script>var wts = new Array', flags=re.S)
            content1 = re1.findall(response_txt)
            if len(content1) == 0:
                print(f'this i is {i}, it has no content')
                continue
            if len(content1) > 1:
                print("stop at ", i)
                break

            logfile.write(f"i: {i}\n")
            re_split = re.compile(r'onclick="updateGraph\(\d+\);">', flags=re.S)
            content2 = re_split.split(content1[0])
            re2 = re.compile(r'[""''/<>=0-9a-zA-Z_().,\-:;&#? ]', flags=re.S)
            for j in content2:
                content3 = re2.sub('', j)
                content4 = content3.split('《', 1)
                content4[1] = '《' + content4[1]
                parallel.append(content4)
                logfile.write(str(content4) + '\n')

            if cnt % 50 == 0:
                print(cnt, content4)
                time.sleep(5)
                if cnt % 100 == 0:
                    time.sleep(60)
                if cnt % 1000 == 0:
                    time.sleep(120)
                if len(content3) == 0:
                    cnt2 += 1
                    if cnt2 == 4:
                        break
            print(f'this i is {i}')

    df_parallel = pd.DataFrame(parallel, columns=['原句', '平行段落'])

    try:
        writer = pd.ExcelWriter(r"Crawler/ctext平行段落.xlsx", mode="a", engine="openpyxl")
        df_parallel.to_excel(writer, index=False, sheet_name=range_name)
        writer.save()
        writer.close()
    except:
        df_parallel.to_excel(r"Crawler/ctext平行段落.xlsx", sheet_name=range_name, index=False)


for range_name, range_list in range_dict.items():
    print(range_name, range_list)
    get_content(range_name, range_list)
