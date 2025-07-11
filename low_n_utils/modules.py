import os
import random
import tensorflow as tf

import numpy as np
import pandas as pd

from low_n_utils import net_MLP as net
from low_n_utils import A003_common
from low_n_utils import A006_common
# import ga
# import predict_common as pdcom

class ParameterInitialization():
    def __init__(self, config, args):
        self.UNIREP_BATCH_SIZE = config.UNIREP_BATCH_SIZE
        self.TOP_MODEL_ENSEMBLE_NMEMBERS = config.TOP_MODEL_ENSEMBLE_NMEMBERS
        self.TOP_MODEL_SUBSPACE_PROPORTION = config.TOP_MODEL_SUBSPACE_PROPORTION
        self.TOP_MODEL_NORMALIZE = config.TOP_MODEL_NORMALIZE
        self.TOP_MODEL_PVAL_CUTOFF = config.TOP_MODEL_PVAL_CUTOFF
        self.SIM_ANNEAL_K = config.SIM_ANNEAL_K
        self.SIM_ANNEAL_INIT_SEQ_MUT_RADIUS = config.SIM_ANNEAL_INIT_SEQ_MUT_RADIUS

        self.seed = args.seed
        self.full_model_name = args.model_name
        self.model_name = self.full_model_name.split("-")[0]
        self.use_bright = args.use_bright
        self.training_objectives = args.training_objectives
        self.top_model_name = args.top_model_name
        self.sampling_method = args.sampling_method
        self.n_train_seqs = args.n_train_seqs

        self.data_set_path = config.DATA_SET_PATH
        self.train_rep_src = config.PROGRAM_CONTROL['train_rep_src']
        self.test_rep_src = config.PROGRAM_CONTROL['test_rep_src']
        self.design_method = config.PROGRAM_CONTROL["design_method"]
        self.pts_model_path = config.PARAMETER['pts_model_path']
        self.n_chains = config.PARAMETER["n_chains"]
        self.T_max = config.PARAMETER["T_max"]
        self.sa_n_iter = config.PARAMETER["sa_n_iter"]
        self.temp_decay_rate = config.PARAMETER["temp_decay_rate"]
        self.nmut_threshold = config.PARAMETER["nmut_threshold"]
        self.exp_name = config.PARAMETER['exp_name']
        self.pts_load_model_method = config.PARAMETER['load_model_method']
        self.output_dir = config.PARAMETER['output_dir']
        self.pdb_path = config.PARAMETER['pdb_path']

        if 'pet' in self.training_objectives:
            self.seq_start, self.seq_end = config.PET_SEQ_INFORMATION['seq_start'], config.PET_SEQ_INFORMATION['seq_end']
            self.reference_seq = config.PET_SEQ_INFORMATION["wt_seq"]
            self.max_mut_pos = config.PET_SEQ_INFORMATION["max_pos"]
            self.min_mut_pos = config.PET_SEQ_INFORMATION["min_pos"]
        elif 'gfp' in self.training_objectives:
            self.seq_start, self.seq_end = config.GFP_SEQ_INFORMATION['seq_start'], config.GFP_SEQ_INFORMATION['seq_end']
            self.reference_seq = config.GFP_SEQ_INFORMATION["wt_seq"]
            self.max_mut_pos = config.GFP_SEQ_INFORMATION["max_pos"]
            self.min_mut_pos = config.GFP_SEQ_INFORMATION["min_pos"]
        
        

        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.init_seqs = A006_common.propose_seqs(
                [self.reference_seq]*self.n_chains, 
                [self.SIM_ANNEAL_INIT_SEQ_MUT_RADIUS]*self.n_chains, 
                min_pos=A006_common.GFP_LIB_REGION[0], 
                max_pos=A006_common.GFP_LIB_REGION[1])
        self.mu_muts_per_seq = 1.5*np.random.rand(self.n_chains) + 1
        print('mu_muts_per_seq:', self.mu_muts_per_seq) # debug
    
    def create_dir_not_exist(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

class SelectLowNDataFram(ParameterInitialization):
    def __init__(self, config, args):
        super(SelectLowNDataFram, self).__init__(config, args)
        if 'fixed' in self.sampling_method:
            if "1-3" in self.sampling_method:
                self.training_set_file = f"{config.FIXED_TRAIN_CSV_PATH}/{str(self.n_train_seqs)}_1-3/train_{str(self.n_train_seqs)}_"
            elif "1-2" in self.sampling_method:
                self.training_set_file = f"{config.FIXED_TRAIN_CSV_PATH}/{str(self.n_train_seqs)}_1-2/train_{str(self.n_train_seqs)}_"
            else:
                self.training_set_file = f"{config.FIXED_TRAIN_CSV_PATH}/{str(self.n_train_seqs)}/train_{str(self.n_train_seqs)}_"
                # self.training_set_file = config.TRAIN_CSV_DICT_FIXED[f'{self.training_objectives}_{str(self.n_train_seqs)}']
        else:
            self.training_set_file = config.TRAIN_CSV_DICT[self.training_objectives]

        if args.do_test == 'True':
            if 'double' in self.training_objectives:
                self.test_set_file_activity = config.TEST_CSV_DICT['pet_activity']
                self.test_set_file_Residual_activity = config.TEST_CSV_DICT['pet_Residual_activity']
                self.test_set_file = None
            else:
                self.test_set_file = config.TEST_CSV_DICT[self.training_objectives]
        else:
            self.test_set_file = None

        
        if 'pet' in self.training_objectives:
            if 'stability' in self.training_objectives:
                print("pet_stability")
                self.fitness_name = 'stability'#选择是进行GFP任务还是稳定性任务
            elif 'activity' in self.training_objectives:
                self.fitness_name = 'activity'
        elif 'gfp' in self.training_objectives:
            self.fitness_name = 'quantitative_function'

    def select_train_df(self):
        if 'fixed' in self.sampling_method:
            sub_train_df = pd.read_csv(f'{self.training_set_file}{self.seed}.csv') 
        else:
            if self.use_bright:
                train_df_all = pd.read_csv(self.training_set_file)
                # train_df_all.drop(train_df_all.columns[[0]], axis=1,inplace=True)
                train_df = train_df_all[train_df_all["quantitative_function"] >= 0.6]
            else:
                train_df = pd.read_csv(self.training_set_file)
                # train_df.drop(train_df.columns[[0]], axis=1,inplace=True)
            # train_df = pd.read_csv(training_set_file)
            if self.sampling_method == 'random':
                print('n_train_seqs:', self.n_train_seqs)
                if 'stability' in self.fitness_name:
                    sub_train_df = train_df[train_df[self.fitness_name].notnull()].sample(n = self.n_train_seqs)
                else:
                    sub_train_df = train_df.sample(n=self.n_train_seqs)
            elif self.sampling_method == 'random_1-3':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df = train_df[train_df["distance"] <= 3].sample(n=self.n_train_seqs)
            elif self.sampling_method == 'random_1-2':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df = train_df[train_df["distance"] <= 2].sample(n=self.n_train_seqs)
                print(sub_train_df.index.values)
            elif self.sampling_method == 'random_1':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df = train_df[train_df["distance"] <= 1].sample(n=self.n_train_seqs)
                print(sub_train_df.index.values)
            elif self.sampling_method == 'random_2':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df = train_df[train_df["distance"] == 2].sample(n=self.n_train_seqs)
                print(sub_train_df.index.values)
            elif self.sampling_method == 'random_3':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df = train_df[train_df["distance"] == 3].sample(n=self.n_train_seqs)
                print(sub_train_df.index.values)
            elif self.sampling_method == 'random_2-3':
                print('n_train_seqs:', self.n_train_seqs)
                print(self.sampling_method)
                sub_train_df_2 = train_df[train_df["distance"] == 2]
                sub_train_df_3 = train_df[train_df["distance"] == 3]
                sub_train_df = pd.concat([sub_train_df_2, sub_train_df_3]).sample(n=self.n_train_seqs)
                print(sub_train_df.index.values)
            elif self.sampling_method == 'all':
                if 'stability' in self.fitness_name:
                    sub_train_df = train_df[train_df[self.fitness_name].notnull()]
                    print(type(sub_train_df))
                    print('train_num:' ,len(list(sub_train_df['name'])))
                else:
                    sub_train_df = train_df
            elif self.sampling_method == '50-50_1.0':
                sub_train_df1 = train_df[train_df[self.fitness_name] >= 1]
                sub_train_df2 = train_df[train_df[self.fitness_name] < 1].sample(n=len(sub_train_df1))
                sub_train_df = pd.concat([sub_train_df1, sub_train_df2])
            elif self.sampling_method == "positive":
                if 'stability' in self.fitness_name:
                    sub_train_df = train_df[train_df['stability'].notnull() >= 1]
                    print(type(sub_train_df))
                    print('train_num:' ,len(list(sub_train_df['name'])))
                else:
                    sub_train_df = train_df[train_df[self.fitness_name] >= 1]
            else:
                raise NameError('Wrong value of `sampling_method`')
        return sub_train_df

    def select_test_df(self, split_num = 96):
        np.random.seed(0)
        random.seed(0)
        if "test_1-3" in self.training_objectives:
            df_1 = pd.read_csv(self.test_set_file)
            df_1.drop(df_1.columns[[0]], axis=1,inplace=True)
            df = df_1[df_1["distance"] <= 3]
        elif "test_3" in self.training_objectives:
            df_1 = pd.read_csv(self.test_set_file)
            df_1.drop(df_1.columns[[0]], axis=1,inplace=True)
            df = df_1[df_1["distance"] == 3]
        else:
            df = pd.read_csv(self.test_set_file)
            # df.drop(df.columns[[0]], axis=1,inplace=True)
        if "split" in self.training_objectives:
            if "SN" in self.training_objectives:
                df_high_function = df[df[self.fitness_name] > 1.5].sample(n=split_num)
                df_low_function = df[df[self.fitness_name] < 0.7]
            elif "SK" in self.training_objectives:
                df_high_function = df[df[self.fitness_name] > 1.0].sample(n=split_num)
                df_low_function = df[df[self.fitness_name] < 0.6]
            test_df = pd.concat([df_high_function, df_low_function])
            print(test_df.head())
            print(len(test_df['name']))
        else:
            test_df = df
        return test_df


    def get_qfunc(self, sub_df):
        qfunc = np.array(sub_df[self.fitness_name])
        return qfunc



class GenerateTopModel(ParameterInitialization):
    def __init__(self, config, args):
        super(GenerateTopModel, self).__init__(config, args)
        
    def sele_top_model(self, train_reps, train_qfunc):
        if self.top_model_name == 'lin':
            print('Building lin top model')
            top_model = A003_common.train_ensembled_ridge(
                    train_reps, 
                    train_qfunc, 
                    n_members=self.TOP_MODEL_ENSEMBLE_NMEMBERS, 
                    subspace_proportion=self.TOP_MODEL_SUBSPACE_PROPORTION,
                    normalize=self.TOP_MODEL_NORMALIZE, 
                    do_sparse_refit=True, 
                    pval_cutoff=self.TOP_MODEL_PVAL_CUTOFF
                )
                
        elif self.top_model_name == "lda":
            top_model = A003_common.train_lda(
                train_reps, 
                train_qfunc
                )

        elif self.top_model_name == 'nn':
            print('Building MLP top model')
            if 'UniRep' in self.model_name:
                if "Augmenting" in self.full_model_name:
                    top_model = net.train_mlp_euni_a(self.seed, train_reps, train_qfunc)
                else:
                    top_model = net.train_mlp_uni(self.seed, train_reps, train_qfunc)
            elif  self.model_name == 'onehot':
                top_model = net.train_mlp_onehot(self.seed, train_reps, train_qfunc)
            else:
                # top_model = net.train_mlp(seed, train_reps, train_qfunc, config)
                top_model = net.train_mlp_Lengthen(self.seed, train_reps, train_qfunc)
        return top_model
    
