import os
import sys
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
# import models
# import unirep
from unirep_hotspot import babbler1900 as babbler
import paths
import models
from data_utils import format_batch_seqs, nonpad_len
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

model_name = 'eUniRep'
csv_path = '/share/jake/hot_spot/data/new_split_set/gfp_3_mutation_sample_0.csv'#'/share/jake/Low_N_data/csv/sn_data_set.csv'
UNIREP_BATCH_SIZE = 1
output_path = f"/share/jake/Low_N_data/ebd/eUniRep/GFP_3mutation_all"

csv_f = pd.read_csv(csv_path)[0:2]
seq_list = list(csv_f["seqs"])
# qfunc_list = list(csv_f["quantitative_function"])
name_list = list(csv_f["name"])

prot_list = []
for i in range(len(name_list)):
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
elif model_name =='OneHot':
    # Just need it to generate one-hot reps.
    # Top model created within OneHotRegressionModel doesn't actually get used.
    base_model = models.OneHotRegressionModel('EnsembledRidge') 
else:
    assert False, 'Unsupported base model'

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.compat.v1.Session(config=config) as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     if 'babbler1900' == base_model.__class__.__name__:
#         hiddens = [] 
#         k = len(seq_list) // UNIREP_BATCH_SIZE
#         if (len(seq_list) % UNIREP_BATCH_SIZE) > 0:
#             k += 1
#         for i in tqdm(range(k)):
#             seq_list_k = seq_list[i*UNIREP_BATCH_SIZE : (i+1)*UNIREP_BATCH_SIZE]
#             hidden_batch = base_model.get_all_hiddens(seq_list_k, sess)
#             hiddens += hidden_batch
        
#         rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
# print(rep.shape)
if 'babbler1900' == base_model.__class__.__name__:
    final_hidden_op, x_ph, batch_size_ph, seq_len_ph, init_state_ph = base_model.get_rep_ops()
    logits_op, loss_op, x_ph, y_ph, batch_size_ph, init_state_ph = base_model.get_babbler_ops()
    batch_size = UNIREP_BATCH_SIZE
    batch_loss_op = base_model.batch_losses
    loss_vals = []

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_batches = int(len(seq_list) / batch_size)
        leftover = len(seq_list) % batch_size
        print("leftover", leftover)
        n_batches += int(bool(leftover))
        for i in tqdm(range(n_batches)):
            # print('----Running inference for batch # %d------' % i)
            if i == n_batches - 1:
                batch_seqs = seq_list[-batch_size:]
            else:
                batch_seqs = seq_list[i*batch_size:(i+1)*batch_size]
            batch_seqs = [seq.replace('-', 'X') for seq in batch_seqs]
            print(batch_seqs)
            # print("batch_seqs: ", batch_seqs)
            batch = format_batch_seqs(batch_seqs)#batch is a list, len(batch) = batch_num, len(batch[0]) = seq_len.
            length = nonpad_len(batch)
            # Run final hidden op
            loss = base_model.get_all_loss(batch_seqs, batch[:, 1:], sess)
            print(loss)
#             loss_ = sess.run(
#                 batch_loss_op,
#                 feed_dict={
#                     # Important! Shift input and expected target by 1.
#                     x_ph: batch[:, :-1],#x_pt 是输入序列的第0为到倒数第2位
#                     y_ph: batch[:, 1:],#y_pt 是输入序列的第1位到最后一位, 它与x_ph错开一位
#                     batch_size_ph: batch.shape[0],
#                     seq_len_ph: length,
#                     init_state_ph:base_model._zero_state
#                 })
#             print(loss_)
#             if i == n_batches - 1:
#                 loss_vals.append(loss_[-leftover:])
#             else:
#                 loss_vals.append(loss_)
#     loss_vals = np.concatenate(loss_vals, axis=0)
#     yhat = list(loss_vals)

# csv_f[f"{model_name}_hotspot_inference"] = yhat
# print(csv_f)


    
