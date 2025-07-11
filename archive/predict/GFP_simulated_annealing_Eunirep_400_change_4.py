import os
import sys
import tensorflow as tf
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing')
sys.path.append('/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing/hyperborg')
import load_file_1 as lf

# task_name_list = ['ePtsRep']# ['eUniRep','UniRep','Random_UniRep','ePtsRep','PtsRep','Random_PtsRep']
# n_train_seqs_list = [400, 2000, 96]
task_name = 'ePtsRep'
train_num = '0096'
n_train_seqs = 400
use_25000_train_seqs = 'False'
train_name_path = '/home/wangqihan/Low_N_test/UniRep_cv/cv_'
real_value_path = '/share/joseph/seqtonpy/gfp/gfp.txt'#/share/seqtonpy/gfp/gfp.txt
task_path = '/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing/hyperborg'
seq_name_path = '/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt'
# real_value_path = "/share/seqtonpy/gfp/gfp.txt"
test_list_path = '/home/wangqihan/ne_sa.txt'
test_list_path_25000 = '/share/joseph/seqtonpy/gfp/test/set_test.txt'
choose_test_data = 'Global'
test_data_separate = 'False'
data_save ='False'
output_path = '/home/caiyi/'

ebd_path_dict = {'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/evoptsrep_af',
'PtsRep':'/share/joseph/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor',
'Random_PtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/random_ptsrep', 'eUniRep':'/home/wangqihan/Low-N_test_ebd/eUniRep',
'UniRep':'/home/wangqihan/Low-N_test_ebd/UniRep', 'Random_UniRep':'/home/wangqihan/Low-N_test_ebd/Random_UniRep'}
test_ebd_path = {'eUniRep':'/home/wangqihan/Eunirep_ebd','UniRep':'/home/wangqihan/unirep_ebd',
'Random_UniRep':'/home/wangqihan/unirep_random_ebd','OneHot':'/home/wangqihan/onehot_ebd',
'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/sa_evoptsrep_af_2e-3','PtsRep':'/home/caiyi/data/gfp/sa_ptsrep',
'Random_PtsRep':'/home/caiyi/embed/gfp/sa_random_ptsrep'}
# for task_name in task_name_list:
#     for n_train_seqs in n_train_seqs_list:
'''### CONFIGURATION ###'''
'''根据任务名称分别载入5个config文件'''
config_list = lf.config_list(task_name,task_path,train_num)
print(config_list)
if use_25000_train_seqs == 'True':
    config,seed,n_train_seqs,model_name,n_chains,T_max,sa_n_iter,temp_decay_rate,min_mut_pos,max_mut_pos,nmut_threshold,output_file = lf.load_config(config_list[0],task_path,n_train_seqs)
    UNIREP_BATCH_SIZE = 400#3500
    TOP_MODEL_DO_SPARSE_REFIT = config.get('sparse_refit', True)
    for num in range(5):
        dic,train_qfunc,init_seqs,mu_muts_per_seq = lf.select_train_seq_25000(train_name_path,n_train_seqs,num,real_value_path,n_chains)
        train_reps = lf.use_seq_2500(num,task_name,train_name_path,n_train_seqs,ebd_path_dict)
        top_model = lf.top_model(train_reps,train_qfunc,TOP_MODEL_DO_SPARSE_REFIT)
        if choose_test_data == '25000':
            yhat,d1,d = lf.get_predict_data_25000(task_name,ebd_path_dict,test_list_path_25000,top_model,real_value_path,num)
            lf.data_save_25000(d1,d,data_save,num,task_name,n_train_seqs,output_path)
        else:
            yhat,d1,d = lf.get_predict_data(task_name,test_ebd_path,top_model,test_list_path,choose_test_data,test_data_separate)
            lf.data_save(d1,d,data_save,test_data_separate,seed,output_path,task_name,n_train_seqs,choose_test_data,train_num)

else:
    for config_name in config_list:
        '''载入config文件，传入超参'''
        config,seed,n_train_seqs,model_name,n_chains,T_max,sa_n_iter,temp_decay_rate,min_mut_pos,max_mut_pos,nmut_threshold,output_file = lf.load_config(config_name,task_path,n_train_seqs)

        for k in config:
            if k == 'init_seqs':
                print(k + ':', config[k][:5])
            else:
                print(k + ':', config[k])

        '''Hard constants'''
        UNIREP_BATCH_SIZE = 400#3500
        TOP_MODEL_DO_SPARSE_REFIT = config.get('sparse_refit', True)
        '''下面这句临时删掉
        assert n_chains <= UNIREP_BATCH_SIZE'''

        '''根据超参筛选出一定数量的训练集序列，并且生成初始的设计序列'''
        train_seqs, train_qfunc, init_seqs, mu_muts_per_seq = lf.selece_train_seq(seed,n_chains,n_train_seqs)

        '''根据超参选择做ebd的模型'''
        print('Setting up base model')
        tf.reset_default_graph()
        if task_name == 'eUniRep' or task_name == 'Random_UniRep' or task_name == 'OneHot':
            base_model = lf.eUnirep_model(model_name,UNIREP_BATCH_SIZE = UNIREP_BATCH_SIZE)
            train_reps,top_model = lf.top_model_unirep(base_model,train_seqs,train_qfunc,nmut_threshold,TOP_MODEL_DO_SPARSE_REFIT,
            UNIREP_BATCH_SIZE = UNIREP_BATCH_SIZE)
            # np.save('/home/wangqihan/Low_N_test/train_reps_old.npy', train_reps)
            # print(a)
        elif task_name == 'UniRep':
            base_model = lf.UniRep_model(UNIREP_BATCH_SIZE = UNIREP_BATCH_SIZE) 
            train_reps,top_model = lf.top_model_unirep(base_model,train_seqs,train_qfunc,nmut_threshold,TOP_MODEL_DO_SPARSE_REFIT,
            UNIREP_BATCH_SIZE = UNIREP_BATCH_SIZE)
        else:
            train_reps = lf.PtsRep_model(task_name,seq_name_path,ebd_path_dict,train_seqs)
            top_model = lf.top_model(train_reps,train_qfunc,TOP_MODEL_DO_SPARSE_REFIT)

        '''得出预测值yhat'''
        if choose_test_data == '25000':
            yhat,d1,d = lf.get_predict_data_25000(task_name,ebd_path_dict,test_list_path_25000,top_model,real_value_path,seed)
            lf.data_save_25000(d1,d,data_save,seed,task_name,n_train_seqs,output_path)
        else:
            yhat,d1,d = lf.get_predict_data(task_name,test_ebd_path,top_model,test_list_path,choose_test_data,test_data_separate)
            lf.data_save(d1,d,data_save,test_data_separate,seed,output_path,task_name,n_train_seqs,choose_test_data,train_num)