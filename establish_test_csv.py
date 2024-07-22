import os
import csv
import pandas as pd


rows = []
datas_2 = []
f1 = open("/home/wangqihan/low-N-protein-engineering-master/泛化验证/split_text_list.txt")
datas_1 = f1.readlines()
for data_1 in datas_1:
    datas_2.append(data_1.split()[0])
with open("/share/joseph/seqtonpy/gfp/gfp.txt", "r")as f:
    datas = f.readlines()
    for data in datas:
        if data.split()[0] + ".npy" in datas_2:
            data_1 = (int(data.split()[0]), float(data.split()[1]))
            rows.append(data_1)
f1.close()
c = open("/home/wangqihan/low-N-protein-engineering-master/泛化验证/old/sk_split_test_set.csv", "w", encoding="utf8")
writer = csv.writer(c)
writer.writerow(['name', 'quantitative_function'])
writer.writerows(rows)
