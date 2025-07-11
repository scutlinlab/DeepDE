import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import tensorflow as tf
import numpy as np
import json
sys.path.append('/home/wangqihan/low-N-protein-engineering-master/泛化验证')
import load_file_4 as lf

task_name_list = ['ePtsRep','PtsRep','Random_PtsRep']#['eUniRep','UniRep','Random_UniRep','ePtsRep','PtsRep','Random_PtsRep']
# 为方便一次性测试多个任务而设置的列表，里面含有多个上游任务名
n_train_seqs_list = [24, 96, 400, 2000]
# 为了方便测试而设置的列表，里面含有训练集数据
choose_test_data_list = ['ALL', '25000']
# 为了方便选择测试集而设置的列表
task_name = 'ePtsRep'
# 上游模型的名字，总共有'eUniRep','UniRep','Random_UniRep','ePtsRep','PtsRep','Random_PtsRep'六个不同的任务。
top_model_name = 'nn'
# 选择下游的模型，有两种选择，'nn'为神经网络模型，'lin'为线性回归模型
train_num = '0024'
# train_num和task_name组合起来可以选择超参文件
n_train_seqs = 24
# 选择训练集的数量，当选择24或96作为训练集时n_train_seqs的值需要与train_num保持一至，当选择其他数量训练集时会根据train_num制定超参文件
use_25000_train_seqs = 'False'
# 这是换用另一种数据集的开关，其为True时训练集和测试集会选用50000数据集中突变数量为1-3的蛋白质。
train_name_path = "/home/wangqihan/Low_N_test/UniRep_cv/cv_"
# 当use_25000_train_seqs为True时,train_name_path提供训练集的文件名。
real_value_path = '/share/joseph/seqtonpy/gfp/gfp.txt'
# 当use_25000_train_seqs为True时,该变量记录全部蛋白质的真实值。
task_path = '/home/wangqihan/low-N-protein-engineering-master/analysis/A006_simulated_annealing/hyperborg'
# 该变量记录超参文件的位置。
seq_name_path = '/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt'
# 该变量记录Low-N原有训练集序列与实验室已知的序列的对应关系，以用于PtsRep模型的训练使用。
test_list_path = '/home/wangqihan/ne_sa.txt'
# 这个变量代表所有Low-N设计出来的（13000左右）序列与其真实值和模型预测值的对应。
choose_test_data = 'SN_split'
# 用于选择不同的测试集，可选的测试集有：ALL,Global,Global_24,Global_96,Random_24,Random_96,Onehot_24,Onehot_96,34536,34536_split,25000(使用50000数据集里面的4-15突变部分)
test_data_separate = 'False'
# 是否将不同模型的测试集分开计算（现在一般不需要分开）
data_save = 'True'
# 是否需要储存画召回曲线的文件
output_path = '/home/wangqihan/'
# 输出文件的地址，输出文件包括指标文件（.txt）和召回曲线画图文件两部分。
recell_output_path = '/home/wangqihan/Low_N_test/画图文件/泛化测试_SN/'
# 输出召回曲线文件的位置。

ebd_path_dict = {'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/evoptsrep_2e-3',
'PtsRep':'/share/joseph/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor',
'Random_PtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/random_ptsrep', 'eUniRep':'/home/wangqihan/Low-N_test_ebd/eUniRep',
'UniRep':'/home/wangqihan/Low-N_test_ebd/UniRep', 'Random_UniRep':'/home/wangqihan/Low-N_test_ebd/Random_UniRep'}
# 所有模型对应的50000数据集embedding，包含训练集和测试集。
test_ebd_path = {'eUniRep':'/home/wangqihan/Eunirep_ebd','UniRep':'/home/wangqihan/unirep_ebd',
'Random_UniRep':'/home/wangqihan/unirep_random_ebd','OneHot':'/home/wangqihan/onehot_ebd',
'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/sa_evoptsrep_af_2e-3','PtsRep':'/home/caiyi/data/gfp/sa_ptsrep',
'Random_PtsRep':'/home/caiyi/embed/gfp/sa_random_ptsrep'}
# 所有模型的13000（Low-N设计的）测试集数据。
real_value_dict = {'25000': '/share/joseph/seqtonpy/gfp/gfp.txt', '34536': '/share/joseph/seqtonpy/gfp/gfp.txt', 
'34536_split': '/share/joseph/seqtonpy/gfp/gfp.txt', 'SN': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_sn.txt',
'FP': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_fp.txt',
'SN_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_sn.txt',
'FP_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/name_to_qfuc/gfp_fp.txt'}
test_ebd_path_SN = {'ePtsRep':'/share/jake/av_GFP_ebd/ePtsRep',
'PtsRep':'/share/jake/av_GFP_ebd/PtsRep','Random_PtsRep':'/share/jake/av_GFP_ebd/RPtsRep', 
'eUniRep':'/share/jake/av_GFP_ebd/SN/eUniRep','UniRep':'/share/jake/av_GFP_ebd/SN/UniRep',
'Random_UniRep':'/share/jake/av_GFP_ebd/SN/RUniRep'}
test_ebd_path_FP = {}
choose_ebd = {'25000': ebd_path_dict, '34536': ebd_path_dict, '34536_split': ebd_path_dict, 'SN': test_ebd_path_SN,
'FP': test_ebd_path_FP, 'ALL': test_ebd_path, 'SN_split': test_ebd_path_SN,
'FP_split': test_ebd_path_FP}
test_list_dict = {'25000': "/share/joseph/seqtonpy/gfp/test/set_test.txt", '34536': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list.txt',
'34536_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/split_text_list.txt',
'SN_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_sn.txt',
'FP_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_fp.txt'}

with open('/home/wangqihan/low-N-protein-engineering-master/Low-N_new/configs/predict_config_gfp_nn.json') as f:
    config_1 = json.load(f)

# 根据任务名称分别载入5个config文件
for task_name in task_name_list:
    for n_train_seqs in n_train_seqs_list:
                # for choose_test_data in  choose_test_data_list:
        config_list = lf.config_list(task_name,task_path,train_num)
        # config_list函数可以根据不同的任务名选择不同的config(超参文件)文件
        print(config_list)
        for config_name in config_list:
            # 测试集选用50000数据集里面的1-3突变序列（约25000个）
            config,seed,n_train_seqs,model_name,n_chains,T_max,sa_n_iter,temp_decay_rate,\
                min_mut_pos,max_mut_pos,nmut_threshold,output_file = lf.load_config(config_name,task_path,n_train_seqs)
            # load_config函数可以根据选择的config文件来为模型传入参数。
            UNIREP_BATCH_SIZE = 400#3500
            TOP_MODEL_DO_SPARSE_REFIT = config.get('sparse_refit', True)
            # 为五折交叉验证提供计数
            train_reps, train_qfunc, init_seqs, mu_muts_per_seq = lf.select_train_seq(top_model_name,task_name,use_25000_train_seqs,real_value_path,
                train_name_path,seed,n_chains,n_train_seqs,ebd_path_dict,seq_name_path)
            top_model = lf.top_model(seed,task_name, top_model_name,config_1,train_reps,train_qfunc,TOP_MODEL_DO_SPARSE_REFIT)
            yhat,d1,d = lf.get_predict_data(top_model_name,config_1,task_name, real_value_dict, test_list_dict, choose_ebd,seed,
                top_model,test_list_path,choose_test_data,test_data_separate)
            lf.data_save(use_25000_train_seqs,d1,d,data_save,test_data_separate,seed,output_path,task_name,
                n_train_seqs,choose_test_data,train_num,top_model_name,recell_output_path)