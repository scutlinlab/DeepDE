import os
import sys
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
# import models
# import unirep
from unirep import babbler1900 as babbler
import paths
import models
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# csv_path = '/home/wangqihan/low-N-protein-engineering-master/data/s3/chip_1/A052b_Chip_1_inferred_brightness_v2.csv'
# output_path = '/share/jake/av_GFP_ebd/SN/UniRep'
model_name = 'eUniRep'
UNIREP_BATCH_SIZE = 1

csv_path = '/share/jake/sk_tumple.csv'#'/share/jake/Low_N_data/csv/sn_data_set.csv'
UNIREP_BATCH_SIZE = 1
output_path = f"/share/jake/ebd/tumple"

seq_list = []
qfuc_list = []
prot_list = []

csv_f = pd.read_csv(csv_path)
seq_list = list(csv_f["seq"])
qfunc_list = list(csv_f["quantitative_function"])
name_list = list(csv_f["name"])

for i in range(len(name_list)):
    if qfunc_list[i]  != qfunc_list[i]:
        continue
    else:
        prot_bro = (name_list[i], list(seq_list[i]))
        prot_list.append(prot_bro)
prot_dic = dict(prot_list)

# base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
if model_name == 'eUniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
elif model_name == 'ET_Global_Init_2':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
elif model_name == 'RUniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
elif model_name == 'UniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path='/home/caiyi/github/unirep-master/1900_weights')
    print('Loading weights from:', '/home/caiyi/github/unirep-master/1900_weights')
elif model_name == 'eUniRep_gfp_new': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031001_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_5000': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031602_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_20000': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031603_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_5000_1': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031601_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_13687': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22032702_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name =='OneHot':
    # Just need it to generate one-hot reps.
    # Top model created within OneHotRegressionModel doesn't actually get used.
    base_model = models.OneHotRegressionModel('EnsembledRidge') 
else:
    assert False, 'Unsupported base model'
# print('csv_id_is:', csv_f['id'][0])
# seq_list = ["".join(prot_dic[csv_f['id'][0]])]
# f1 = open('/home/wangqihan/ne_sa.txt','a')
# with open('/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/sa.txt') as f:
#     lines = f.readlines()
# for i in tqdm(lines):
#     name_txt = i.split("\t")[0]
#     if name_txt in name_list:
#         f1.write(i)
# f1.close()
# f.close()
# f2 = open('/home/wangqihan/ne_sa.txt','r')
# lines2 = f2.readlines()
# assert len(lines2) == len(name_list)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(len(name_list))
    for name in tqdm(name_list):
            # print(name)
        # if name.startswith('GFP_SimAnneal-'):
            seq_list = ["".join(prot_dic[name])]
            # print(seq_list)
            if 'babbler1900' == base_model.__class__.__name__:
                assert len(seq_list) <= UNIREP_BATCH_SIZE
                # print(len(seq_list[0]))
                seq_list_batch = seq_list
                hidden_batch = base_model.get_all_hiddens(seq_list_batch,sess)
                # print(hidden_batch.shape)
                rep_batch = np.stack([np.mean(s, axis=0) for s in hidden_batch],0)
                # rep_batch = np.stack(hidden_batch,0)
                # print(rep_batch.shape)
            elif 'OneHotRegressionModel' == base_model.__class__.__name__:
                # print('1')
                rep_batch = base_model.encode_seqs(seq_list)
            # print(rep_batch.shape)
            # print(type(rep_batch))
            np.save(output_path + '/' + str(name) + '.npy',rep_batch[0])
            # print(type(rep_batch[0]))
# print(kdjfklasdj)
seq_ebd = np.load(output_path + '/' + str(name) + '.npy')
print(seq_ebd)
print(seq_ebd.shape)
# print(os.listdir(output_path))

    
