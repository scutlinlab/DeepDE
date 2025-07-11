import os
from tkinter import E
import numpy as np
from low_n_utils import paths
from low_n_utils import models, A003_common

TOP_MODEL_ENSEMBLE_NMEMBERS = models.ENSEMBLED_RIDGE_PARAMS['n_members']
TOP_MODEL_SUBSPACE_PROPORTION = models.ENSEMBLED_RIDGE_PARAMS['subspace_proportion']
TOP_MODEL_NORMALIZE = models.ENSEMBLED_RIDGE_PARAMS['normalize']
TOP_MODEL_PVAL_CUTOFF = models.ENSEMBLED_RIDGE_PARAMS['pval_cutoff']
SIM_ANNEAL_K = 1
SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = 3

DATA_SET_PATH = "/share/jake/github/low-n_data"

OUTPUT_PATH = "/share/jake/github/low_n_output"

PCA_19_EBD_PATH = "/share/jake/github/low-n_data/pca/pca-19.npy"

TOP_MODEL_PATH = '/share/jake/github/low_n_output/method_2/top_model/lin/top_model_eUniRep-Augmenting_random_1-2_1000.pkl'

EBD_PATH_GFP_50000 = {'ePtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/evoptsrep_2e-3',
    'PtsRep':'/share/joseph/seqtonpy/gfp/knn_self_512_full/self_20201216_4__self_20201215_5_sota_right_n2_knnnet150_del_tor',
    'Random_PtsRep':'/home/caiyi/embed/gfp/evo_ptsrep/random_ptsrep', 
    'eUniRep':'/share/jake/Low_N_data/ebd/eUniRep/GFP_sk',#/share/jake/Low_N_data/ebd/GFP_sk
    'UniRep':'/share/jake/Low_N_data/ebd/UniRep/GFP_sk', 
    'Random_UniRep':'/share/jake/Low_N_data/ebd/RUniRep/GFP_sk',
    'eUniRep_gfp_590': '/share/jake/Low_N_data/ebd/eUniRep_gfp_590/GFP_sk',
    'eUniRep_gfp_5000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000/GFP_sk',
    'eUniRep_gfp_20000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_20000/GFP_sk',
    'eUniRep_gfp_5000_1': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000_1/GFP_sk',
    'eUniRep_gfp_13687': '/share/jake/Low_N_data/ebd/eUniRep_gfp_13687/GFP_sk'}
# 所有模型对应的50000数据集embedding，包含训练集和测试集。
EBD_PATH_GFP_13000 = {'eUniRep':'/home/wangqihan/Eunirep_ebd',
    'UniRep':'/home/wangqihan/unirep_ebd',
    'Random_UniRep':'/home/wangqihan/unirep_random_ebd',
    'OneHot':'/home/wangqihan/onehot_ebd',
    'ePtsRep':'/home/caiyi/embed/gfp/sa_ptsrep/',
    'PtsRep':'/home/caiyi/embed/gfp/sa_ptsrep',
    'Random_PtsRep':'/home/caiyi/embed/gfp/sa_random_ptsrep'}
# 所有模型的13000（Low-N设计的）测试集数据。
TEST_CSV_DICT = {'25000': '/share/joseph/seqtonpy/gfp/gfp.txt', 
    'gfp_SK': "/share/jake/Low_N_data/test_csv/sk_test_set_distance.csv", 
    'gfp_SK_split': "/share/jake/Low_N_data/test_csv/sk_test_set_distance.csv",
    'gfp_SK_split_test_1-3': "/share/jake/Low_N_data/test_csv/sk_test_set_distance.csv", 
    'gfp_SK_test_1-3': "/share/jake/Low_N_data/test_csv/sk_test_set_distance.csv",
    'gfp_SK_split_test_3': "/share/jake/Low_N_data/csv/sk_data_set_distance.csv", 
    'gfp_SK_test_3': "/share/jake/Low_N_data/csv/sk_data_set_distance_classify.csv",
    'gfp_SK_test_3_step2_bright': "/share/jake/Low_N_data/test_csv/sk_test_set_mutation3_split2_9200_distance.csv",
    'gfp_SK_step2_bright': "/share/jake/Low_N_data/test_csv/sk_test_set_split2_19948_distance.csv",
    'gfp_SK_test_3_step2_bright_continue': "/share/jake/Low_N_data/test_csv/sk_test_set_mutation3_split2_9200_distance.csv",
    'gfp_SK_step2_bright_continue': "/share/jake/Low_N_data/test_csv/sk_test_set_split2_19948_distance.csv",
    'gfp_SK_drop_half_3mutation': "/share/jake/Low_N_data/test_csv/sk_test_3mutation_half.csv",
    'gfp_SN': '/share/jake/Low_N_data/test_csv/sn_test_set.csv',
    'gfp_FP': '/share/jake/Low_N_data/test_csv/fp_test_set.csv',
    'gfp_SN_split': '/share/jake/Low_N_data/test_csv/sn_test_set.csv',
    'gfp_FP_split': '/share/jake/Low_N_data/test_csv/fp_test_set.csv',
    'gfp_SK_low_predict_hight': '/share/jake/Low_N_data/test_csv/sk_4-25_mutation.csv',
    'gfp_SK_low_predict_hight_split': '/share/jake/Low_N_data/test_csv/sk_4-25_mutation.csv',
    'gfp_fake': "/share/jake/UniRep泛化验证数据/随机生成序列/test_32509.csv",
    'LN_design': "/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/designed_test_set_clear.csv",
    'pet_activity': '/home/caiyi/data/petase/petase_splits_1/acti_test_', #/home/caiyi/data/petase/petase_splits_1/acti_test_,  /home/wangqihan/csv/pet/pet_test_set/artificial_pet_test_set.csv
    'pet_stability': '/home/caiyi/data/petase/petase_splits_1/stab_test_'}

TRAIN_CSV_DICT = {'pet_activity': "/home/caiyi/data/petase/petase_splits_1/acti_train_", #"/home/caiyi/data/petase/petase_splits_1/acti_train_",  /home/wangqihan/csv/pet/pet_train_set/artificial_pet_train_set.csv
    'pet_stability': "/home/caiyi/data/petase/petase_splits_1/stab_train_", 
    '25000': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv", 
    'gfp_SK': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv", 
    'gfp_SK_split': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_SK_split_test_1-3': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv", 
    'gfp_SK_test_1-3': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_SK_split_test_3': "/share/jake/Low_N_data/csv/sk_data_set_distance.csv",
    'gfp_SK_test_3_step2_bright': "/share/jake/Low_N_data/csv/sk_data_set_distance.csv",
    'gfp_SK_test_3': "/share/jake/Low_N_data/csv/sk_data_set_distance.csv",
    'gfp_SK_step2_bright': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_SK_test_3_step2_bright_continue': "/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_0.csv",
    'gfp_SK_step2_bright_continue': "/share/jake/Low_N_data/train_csv/gfp_SK_train_random_1000_0.csv",
    'gfp_SK_drop_half_3mutation': "/share/jake/Low_N_data/train_csv/sk_train_drop_half_3mutation.csv",
    'gfp_SN': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_FP': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv", 
    'LN_design': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv", 
    'gfp_SN_split': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_FP_split': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_fake': "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv",
    'gfp_SK_low_predict_hight': '/share/jake/Low_N_data/train_csv/sk_1-3_mutation.csv',
    'gfp_SK_low_predict_hight_split': '/share/jake/Low_N_data/train_csv/sk_1-3_mutation.csv'}

TRAIN_CSV_DICT_FIXED = {'gfp_SK_24': '/home/wangqihan/new_train_set/24/train_24_',
    'gfp_SK_96': '/home/wangqihan/new_train_set/96/train_96_',
    'gfp_SK_400': '/home/wangqihan/new_train_set/400/train_400_',
    'gfp_SK_1000': '/home/wangqihan/new_train_set/1000/train_1000_',
    'gfp_SK_2000': '/home/wangqihan/new_train_set/2000/train_2000_'}
    
FIXED_TRAIN_CSV_PATH = "/home/wangqihan/new_train_set"

TEST_EBD_PATH_SN = {'ePtsRep': '/share/jake/av_GFP_ebd/ePtsRep',
    'PtsRep':'/share/jake/av_GFP_ebd/PtsRep',
    'Random_PtsRep':'/share/jake/av_GFP_ebd/RPtsRep', 
    'eUniRep':'/share/jake/Low_N_data/ebd/GFP_sn',
    'UniRep':'/share/jake/Low_N_data/ebd/UniRep/GFP_sn',
    'Random_UniRep':'/share/jake/Low_N_data/ebd/RUniRep/GFP_sn',
    'eUniRep_gfp_590': '/share/jake/Low_N_data/ebd/eUniRep_gfp_590/GFP_sn',
    'eUniRep_gfp_5000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000/GFP_sn',
    'eUniRep_gfp_20000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_20000/GFP_sn',
    'eUniRep_gfp_5000_1': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000_1/GFP_sn',
    'eUniRep_gfp_13687': '/share/jake/Low_N_data/ebd/eUniRep_gfp_13687/GFP_sn'}

TEST_EBD_PATH_FP = {'UniRep': '/share/jake/Low_N_data/ebd/UniRep/GFP_fp',
    'eUniRep_gfp_590': '/share/jake/Low_N_data/ebd/eUniRep_gfp_590/GFP_fp',
    'eUniRep_gfp_5000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000/GFP_fp',
    'eUniRep_gfp_20000': '/share/jake/Low_N_data/ebd/eUniRep_gfp_20000/GFP_fp',
    'eUniRep_gfp_5000_1': '/share/jake/Low_N_data/ebd/eUniRep_gfp_5000_1/GFP_fp',
    'eUniRep_gfp_13687': '/share/jake/Low_N_data/ebd/eUniRep_gfp_13687/GFP_fp'}

EBD_PATH_GFP_32509 = {'eUniRep': '/share/jake/Low_N_data/ebd/eUniRep/GFP_32509'}

TEST_EBD_PATH_PET = {'Alphafold': '/data/wangqihan/af_output/pet_ebd', #'/data/wangqihan/af_output/pet_ebd'
    'Openfold': '/data/wangqihan/output/pet_ebd'}

TEST_LIST_DICT = {'25000': "/share/joseph/seqtonpy/gfp/test/set_test.txt", 
    'SK': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list.txt',
    'SK_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/split_text_list.txt',
    'gfp_SN_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_sn.txt',
    'gfp_FP_split': '/home/wangqihan/low-N-protein-engineering-master/泛化验证/text_list/split_text_list_fp.txt'}

CHOOSE_TRAIN_EBD = {'25000': EBD_PATH_GFP_50000, 
    'gfp_SK': EBD_PATH_GFP_50000, 
    'gfp_SK_split': EBD_PATH_GFP_50000,
    'gfp_SK_split_test_1-3': EBD_PATH_GFP_50000,
    'gfp_SK_test_1-3': EBD_PATH_GFP_50000,
    'gfp_SK_split_test_3': EBD_PATH_GFP_50000,
    'gfp_SK_test_3': EBD_PATH_GFP_50000,
    'gfp_SK_test_3_step2_bright': EBD_PATH_GFP_50000,
    'gfp_SK_step2_bright': EBD_PATH_GFP_50000,
    'gfp_SK_test_3_step2_bright_continue': EBD_PATH_GFP_50000,
    'gfp_SK_step2_bright_continue': EBD_PATH_GFP_50000,
    'gfp_SK_drop_half_3mutation': EBD_PATH_GFP_50000,
    'gfp_SN': TEST_EBD_PATH_SN,
    'gfp_FP': TEST_EBD_PATH_FP, 
    'LN_design': EBD_PATH_GFP_13000, 
    'gfp_SN_split': TEST_EBD_PATH_SN,
    'gfp_FP_split': TEST_EBD_PATH_FP, 
    'pet_activity': TEST_EBD_PATH_PET,
    'pet_stability': TEST_EBD_PATH_PET,
    'gfp_SK_low_predict_hight': EBD_PATH_GFP_50000,
    'gfp_SK_low_predict_hight_split': EBD_PATH_GFP_50000}

CHOOSE_TEST_EBD = {'25000': EBD_PATH_GFP_50000, 
    'gfp_SK': EBD_PATH_GFP_50000,
    'gfp_SK_split': EBD_PATH_GFP_50000,
    'gfp_SK_split_test_1-3': EBD_PATH_GFP_50000,
    'gfp_SK_test_1-3': EBD_PATH_GFP_50000,
    'gfp_SK_split_test_3': EBD_PATH_GFP_50000,
    'gfp_SK_test_3': EBD_PATH_GFP_50000,
    'gfp_SK_test_3_step2_bright': EBD_PATH_GFP_50000,
    'gfp_SK_step2_bright': EBD_PATH_GFP_50000,
    'gfp_SK_test_3_step2_bright_continue': EBD_PATH_GFP_50000,
    'gfp_SK_step2_bright_continue': EBD_PATH_GFP_50000,
    'gfp_SK_drop_half_3mutation': EBD_PATH_GFP_50000,
    'gfp_SN': TEST_EBD_PATH_SN,
    'gfp_FP': TEST_EBD_PATH_FP, 
    'gfp_fake': EBD_PATH_GFP_32509,
    'gfp_SK_low_predict_hight': EBD_PATH_GFP_50000,
    'gfp_SK_low_predict_hight_split': EBD_PATH_GFP_50000, 
    'LN_design': EBD_PATH_GFP_13000, 
    'gfp_SN_split': TEST_EBD_PATH_SN,
    'gfp_FP_split': TEST_EBD_PATH_FP, 
    'pet_activity': TEST_EBD_PATH_PET, 
    'pet_stability': TEST_EBD_PATH_PET}

PET_SEQ_INFORMATION = {
    "wt_seq": "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS",
	"seq_start": 28,
	"seq_end": 290,
	"struct_seq_len": 262,
	"min_pos": 27,
	"max_pos": 289,}

GFP_SEQ_INFORMATION = {
    "wt_seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
	"seq_start": 6,
	"seq_end": -9,
	"struct_seq_len": 223,
	"min_pos": 29,
	"max_pos": 110,}

PROGRAM_CONTROL = {"gpu": 2,
    "seed": 4,
    "model": "eUniRep",
    "training_objectives": "gfp_34536_split", #"gfp_34536_split",
    "top_model_name": "lin",
    "sampling_method": "random_fixed",
    "n_train_seqs": 1000,
    "train_rep_src": "generate",
	"test_rep_src": "generate",#load_by_name, load_by_seq, generate
    "do_design": True, 
    "do_test": True, 
    "save_test_result": True, 
    "design_method": "MCMC",
    "do_predict": False
    }

PARAMETER = {"exp_name": "220106_lin",
    "output_dir": "/share/jake/low_n_output/recall",
	"seq2name_file": "/home/caiyi/github/low-N-protein-engineering-master/analysis/A006_simulated_annealing/gfp_seq2name.txt",
	"pts_model_path": "/home/caiyi/seqrep/outputs/models/ptsrep__20121505_5_sota_right_n2_knnnet150_del_tor/50_model.pth.tar",
	# "pdb_path": "/home/caiyi/data/petase/pdb/6qgc.pdb",
    "pdb_path": "/share/joseph/seqtonpy/gfp/pdb/1emm_mod.pdb",
	"load_model_method": "state_dict",
    "n_chains": 3500,
    "T_max": np.ones(3500,) * 0.01,
    "sa_n_iter": 3000,
    "temp_decay_rate": 1.0,
    "nmut_threshold": 15
	}

UNIREP_BATCH_SIZE = 1000

NET_MLP = {"seq_len": 223,
	"mlp_input_size": 768,
	"mlp_hidden_size": 512,
	"mlp_hidden_act": "relu"}

NET_MLP_1 = {"seq_len": 239,
	"mlp_input_size": 1900,
	"mlp_hidden_size": 512,
	"mlp_hidden_act": "relu"}

NET_LOSS = {"val_set_prop": 0.2,
	"accumulated_iters": 1,
	"train_batch_size": 1,
	"val_batch_size": 100,
	"test_batch_size": 100,
	"learning_rate": 0.001,
	"weight_decay": 0.00001,
	"criterion": "loss",
	"patience": 50,
	"converg_thld": 0}

DO_PREDICT = {"pre_path": "/data/wangqihan/af_output/eUniRep_pet_stability_ga_3000_design", 
    "nseq_select": 100, 
    "save_path": "/home/wangqihan/Low_n_alphafold_test/design/220106_lin_pet_stability_eUniRep_petase_96_GA_0-selected_seqs", 
    "seq_path": "/home/wangqihan/Low_n_alphafold_test/design/design_ga_seqs/eUniRep_pet_0_stability_GA_top3000.txt", 
    "choose_top": True}
 
DEFAULT_PARAMETERS = {}

ALPHAFOLD2 = {"target": "pair"}

OPENFOLD = {"target": "pair"}

