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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.reset_default_graph()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# csv_path = '/home/wangqihan/low-N-protein-engineering-master/data/s3/chip_1/A052b_Chip_1_inferred_brightness_v2.csv'
# output_path = '/share/jake/av_GFP_ebd/SN/UniRep'
model_name = 'eUniRep'
UNIREP_BATCH_SIZE = 1000

input_path = '/share/jake/hot_spot/data/sk_2'#'/share/jake/Low_N_data/csv/sn_data_set.csv'
output_path = f"/share/jake/Low_N_data/ebd/eUniRep/GFP_average_2"
csv_list = os.listdir(input_path)

# base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
if model_name == 'eUniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
elif model_name == 'ET_Global_Init_2':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
elif model_name == 'RUniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
elif model_name == 'UniRep':
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path='/home/caiyi/github/unirep-master/1900_weights', config = tf_config)
    print('Loading weights from:', '/home/caiyi/github/unirep-master/1900_weights')
elif model_name == 'eUniRep_gfp_new': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031001_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_5000': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031602_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_20000': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031603_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_5000_1': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22031601_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name == 'eUniRep_gfp_13687': 
    UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22032702_local_gfp_unirep/'
    print('UNIREP_WEIGHT_PATH: ', UNIREP_WEIGHT_PATH)
    base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
    print('Loading weights from:', UNIREP_WEIGHT_PATH)
elif model_name =='OneHot':
    # Just need it to generate one-hot reps.
    # Top model created within OneHotRegressionModel doesn't actually get used.
    base_model = models.OneHotRegressionModel('EnsembledRidge') 
else:
    assert False, 'Unsupported base model'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for csv_name in csv_list:
        csv_path = f"{input_path}/{csv_name}"
        seq_list = []
        qfuc_list = []
        prot_list = []

        csv_f = pd.read_csv(csv_path)
        seq_list = list(csv_f["seq"])
        # qfunc_list = list(csv_f["quantitative_function"])
        name_list = list(csv_f["name"])
        if 'babbler1900' == base_model.__class__.__name__:
            hiddens = [] 
            k = len(seq_list) // UNIREP_BATCH_SIZE
            if (len(seq_list) % UNIREP_BATCH_SIZE) > 0:
                k += 1
            for i in tqdm(range(k)):
                seq_list_k = seq_list[i*UNIREP_BATCH_SIZE : (i+1)*UNIREP_BATCH_SIZE]
                hidden_batch = base_model.get_all_hiddens(seq_list_k, sess)
                hiddens += hidden_batch
            rep_batch = np.stack([np.mean(s, axis=0) for s in hiddens],0)
        elif 'OneHotRegressionModel' == base_model.__class__.__name__:
            rep_batch = base_model.encode_seqs(seq_list)
        print(rep_batch.shape)
        # print(rep_batch[0][0], rep_batch[100][0], rep_batch[200][0])
        rep_averatge = np.mean(rep_batch, axis=0)
        print(rep_averatge.shape)
        print(rep_batch[10] == rep_averatge)
        # print(type(rep_batch))
        np.save(output_path + '/' + str(csv_name.split(".")[0]) + '.npy',rep_averatge)
        # print(type(rep_batch[0]))
    # print(kdjfklasdj)
    seq_ebd = np.load(output_path + '/' + str(csv_name.split(".")[0]) + '.npy')
    print(seq_ebd)
    print(seq_ebd.shape)
    # print(os.listdir(output_path))

        
