import pickle
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import random
import os
import utils
from low_n_utils import paths
from low_n_utils import models
from torch.utils.data import DataLoader
# from low_n_utils.unirep import babbler1900 as babbler
from low_n_utils.unirep import babbler1900 as babbler
from low_n_utils.unirep_hotspot import babbler1900 as babbler_hotspot
from low_n_utils.process_pdb import process_pdb
from low_n_utils import rep_utils
import Levenshtein
import misc_utils
from low_n_utils import modules
import A006_common
import ga
import net_MLP as net
from misc_utils import PtsRep
from data_utils import format_batch_seqs, nonpad_len

def select_basemodel(seed, model_name, UNIREP_BATCH_SIZE, tf_config):
    np.random.seed(seed)
    random.seed(seed)
    if model_name == 'UniRep':
        UNIREP_WEIGHT_PATH = '/home/caiyi/github/unirep_embedding/1900_weights/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name == 'eUniRep':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH, config = tf_config)
        print(paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    elif model_name == 'eunirep_2':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    elif model_name == 'Random_UniRep':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    elif model_name =='onehot':
        # Just need it to generate one-hot reps.
        # Top model created within OneHotRegressionModel doesn't actually get used.
        base_model = models.OneHotRegressionModel('EnsembledRidge')
    elif model_name == 'eUniRep_uniref100': 
        UNIREP_WEIGHT_PATH = '/share/caiyi/learning-protein-fitness-from-evolutionary-and-assay-labelled-data/unirep_weights/GFP_AEQVI/uniref100/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name == 'eUniRep_833': 
        UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22112511_local_gfp_unirep/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name == 'eUniRep_new_833': 
        UNIREP_WEIGHT_PATH = '/home/wangqihan/unirep/lm__22120122_local_gfp_unirep/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    else:
        assert False, 'Unsupported base model'
    return base_model

def select_basemodel_hotspot(seed, model_name, UNIREP_BATCH_SIZE, tf_config):
    np.random.seed(seed)
    random.seed(seed)
    if model_name == 'UniRep':
        UNIREP_WEIGHT_PATH = '/home/caiyi/github/unirep_embedding/1900_weights/'
        base_model = babbler_hotspot(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name == 'eUniRep':
        base_model = babbler_hotspot(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH, config = tf_config)
        print(paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    elif model_name == 'eunirep_2':
        base_model = babbler_hotspot(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    elif model_name == 'Random_UniRep':
        base_model = babbler_hotspot(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    elif model_name =='onehot':
        # Just need it to generate one-hot reps.
        # Top model created within OneHotRegressionModel doesn't actually get used.
        base_model = models.OneHotRegressionModel('EnsembledRidge')
    elif model_name == 'eUniRep_uniref100': 
        UNIREP_WEIGHT_PATH = '/share/caiyi/learning-protein-fitness-from-evolutionary-and-assay-labelled-data/unirep_weights/GFP_AEQVI/uniref100/'
        base_model = babbler_hotspot(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config = tf_config)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    else:
        assert False, 'Unsupported base model'
    return base_model


class RepInference(modules.ParameterInitialization):

    def __init__(self, config, args):
        super(RepInference, self).__init__(config, args)
        self.target_protein = args.training_objectives.split("_")[0]
        self.test_name = args.training_objectives.split("_")[1]
        self.onehot_base_model = models.OneHotRegressionModel('EnsembledRidge')
        self.aas = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        self.pca19_ebd_path = config.PCA_19_EBD_PATH
    
    def seqs_to_pca(self, seqs):
        pca_19 = np.load(self.pca19_ebd_path)
        aa_dict = {}
        for i, aa in enumerate(self.aas):
            aa_dict[aa] = pca_19[i]
        hiddens = []
        # print(seqs[:2])
        # print(asdjhfgkahgjkadghkahghk)
        for seq in seqs:
            seq_pca = np.array([])
            for i, aa in enumerate(seq):
                seq_pca = np.concatenate((seq_pca, aa_dict[aa]), axis=0)
            hiddens.append(seq_pca)
        return np.array(hiddens)

    def generate_uni_reps(self, dataframe:pd.DataFrame, hotspot = False):
        seq_list = list(dataframe["seq"])
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        if "UniRep" in self.model_name or "onehot" in self.model_name:
            sess = tf.Session(config=tf_config)#config=tf_config
            if hotspot:
                batch_size = 1
                base_model = select_basemodel_hotspot(self.seed, self.model_name, batch_size, tf_config)
            else:
                batch_size = self.UNIREP_BATCH_SIZE
                base_model = select_basemodel(self.seed, self.model_name, batch_size, tf_config)
            sess.run(tf.global_variables_initializer())
        else:
            raise ValueError(
                "can not find the base model"
                                    )
        if 'babbler1900' == base_model.__class__.__name__:
            hiddens = [] 
            k = len(seq_list) // batch_size
            if (len(seq_list) % batch_size) > 0:
                k += 1
            for i in tqdm(range(k)):
                seq_list_k = seq_list[i*batch_size : (i+1)*batch_size]
                hidden_batch = base_model.get_all_hiddens(seq_list_k, sess)
                hiddens += hidden_batch
            if self.top_model_name == 'lin':
                rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            elif self.top_model_name == 'nn':
                rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
                
        elif 'OneHotRegressionModel' == base_model.__class__.__name__:
            rep = base_model.encode_seqs(seq_list)
            
        return rep


    def uni_inf(self, base_model, batch_size, seq_list):
        if 'babbler1900' == base_model.__class__.__name__:
            final_hidden_op, x_ph, batch_size_ph, seq_len_ph, init_state_ph = base_model.get_rep_ops()
            logits_op, loss_op, x_ph, y_ph, batch_size_ph, init_state_ph = base_model.get_babbler_ops()
            if len(seq_list) < self.UNIREP_BATCH_SIZE:
                batch_size = len(seq_list)
            # else:
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
                    # print("batch_seqs: ", batch_seqs)
                    batch = format_batch_seqs(batch_seqs)
                    # print("batch: ",batch)
                    # print(len(batch[0]))
                    length = nonpad_len(batch)
                    # Run final hidden op
                    loss_ = sess.run(
                        batch_loss_op,
                        feed_dict={
                            # Important! Shift input and expected target by 1.
                            x_ph: batch[:, :-1],
                            y_ph: batch[:, 1:],
                            batch_size_ph: batch.shape[0],
                            seq_len_ph: length,
                            init_state_ph:base_model._zero_state
                        })
                    if i == n_batches - 1:
                        loss_vals.append(loss_[-leftover:])
                    else:
                        loss_vals.append(loss_)
            loss_vals = np.concatenate(loss_vals, axis=0)# the shape of loss_vals is (seq_num,)

        return np.expand_dims(loss_vals, 1)

    def generate_uni_inference(self, seq_list):
        inference_seq_num = len(seq_list)
        if self.UNIREP_BATCH_SIZE > inference_seq_num:
            batch_size = inference_seq_num
        else:
            batch_size = self.UNIREP_BATCH_SIZE
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        if "UniRep" in self.model_name:
            base_model = select_basemodel(self.seed, self.model_name, batch_size, tf_config)
        # if "UniRep" in self.model_name or "onehot" in self.model_name:
        #     self.sess = tf.Session(config=tf_config)#config=tf_config
        #     self.base_model = select_basemodel(self.seed, self.model_name, batch_size, tf_config)
        #     self.sess.run(tf.global_variables_initializer())
        loss_vals = self.uni_inf(base_model, batch_size, seq_list)
        return loss_vals #the shape of inference return is (seq_num, 1)

    def load_vae_inference(self, seq_list, inference_path = None):
        if inference_path == None:
            inference_path = os.path.join(self.data_set_path, "csv", self.target_protein, f"{self.test_name.lower()}_data_set_vae_inference.csv")
        all_vae_df = pd.read_csv(inference_path)
        seq2score_dict = dict(zip(list(all_vae_df["seq"]), list(all_vae_df["vae_inference"])))
        scores = np.array([seq2score_dict.get(s.upper(), 0.0) for s in seq_list])# the shape of scores is (seq_num,)
        print(np.expand_dims(scores, 1).shape)
        return np.expand_dims(scores, 1) #the shape of vae inference return is (seq_num, 1)

    def load_uni_inference_method1(self, input_df):
        scores = np.array(list(input_df["max_inference_function"]))
        return np.expand_dims(scores, 1)


    def generate_inference(self, seq_list, vae_inference_path = None):
        if "UniRep" in self.model_name:
            rep_inference = -self.generate_uni_inference(seq_list)
        elif "vae" in self.model_name:
            rep_inference = self.load_vae_inference(seq_list, inference_path = vae_inference_path)
        return rep_inference

    def get_logits(self, seq_list):
        batch_size = self.UNIREP_BATCH_SIZE
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        base_model = select_basemodel(self.seed, self.model_name, batch_size, tf_config)
        sess.run(tf.global_variables_initializer())
  
        if 'babbler1900' == base_model.__class__.__name__:
            logits = [] 
            k = len(seq_list) // self.UNIREP_BATCH_SIZE
            if (len(seq_list) % self.UNIREP_BATCH_SIZE) > 0:
                k += 1
            for i in tqdm(range(k)):
                seq_list_k = seq_list[i*self.UNIREP_BATCH_SIZE : (i+1)*self.UNIREP_BATCH_SIZE]
                hidden_batch, logits_batch = base_model.get_all_hiddens(seq_list_k, sess, return_logits=True)
                logits += logits_batch
                
        return logits

    def Augmenting(self, dataframe:pd.DataFrame, rep_inf = None, use_pca = False):
        if rep_inf == None:
            rep_inference = self.generate_inference(dataframe["seq"]) # the shape of inference is (seq_num, 1)
        else:
            rep_inference = rep_inf
        if "pca" in self.full_model_name:
            reps_ebd = self.seqs_to_pca(list(dataframe["seq"]))
        else:
            reps_ebd = self.onehot_base_model.encode_seqs(list(dataframe["seq"])) # the shape of onthot is (seq_num, 20 * seq_len)
        reps = np.concatenate([reps_ebd, rep_inference], axis=1) # the shape of Augmenting is (seq_num, 20 * seq_len + 1)
        return reps

    def concate(self, dataframe:pd.DataFrame):
        rep_ebd = self.generate_uni_reps(dataframe) # the shape of ebd is (seq_num, 1900)
        reps_oh = self.onehot_base_model.encode_seqs(list(dataframe["seq"])) # the shape of onthot is (seq_num, 20 * seq_len)
        reps = np.concatenate([reps_oh, rep_ebd], axis=1) # the shape of Augmenting is (seq_num, 20 * seq_len + 1900)
        return reps

    
    


class RepManage(modules.ParameterInitialization):

    def __init__(self, config, args):
        super(RepManage, self).__init__(config, args)
        if self.train_rep_src != 'generate':
            self.train_rep_path = config.CHOOSE_TRAIN_EBD[self.training_objectives][self.model_name]
        else:
            self.train_rep_path = None
        
        if self.test_rep_src != 'generate':
            if 'double' in self.training_objectives:
                self.test_rep_path_activity = config.CHOOSE_TEST_EBD['pet_activity'][self.model_name]
                self.test_rep_path_Residual_activity = config.CHOOSE_TEST_EBD['pet_Residual_activity'][self.model_name]
                self.test_rep_path = None
            else:
                self.test_rep_path = config.CHOOSE_TEST_EBD[self.training_objectives][self.model_name]
        else:
            self.test_rep_path = None
            
        self.HYDROPATHICITY = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                               1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
        self.BULKINESS = [11.5, 13.46, 11.68, 13.57, 19.8, 3.4, 13.69, 21.4, 15.71, 21.4,
                          16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03]
        self.FLEXIBILITY = [14.0, 0.05, 12.0, 5.4, 7.5, 23.0, 4.0, 1.6, 1.9, 5.1,
                            0.05, 14.0, 0.05, 4.8, 2.6, 19.0, 9.3, 2.6, 0.05, 0.05]
        self.AA_LIST = ['A', 'C', 'D', 'E', 'F', 
                        'G', 'H', 'I', 'K', 'L', 
                        'M', 'N', 'P', 'Q', 'R', 
                        'S', 'T', 'V', 'W', 'Y']
        self.hydropathicity = (5.5 - np.array(self.HYDROPATHICITY)) / 10
        self.bulkiness = np.array(self.BULKINESS) / 21.67
        self.flexibility = (25 - np.array(self.FLEXIBILITY)) / 25
        # self.device = device
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        if "UniRep" in self.model_name or "onehot" in self.model_name:
            self.sess = tf.Session(config=tf_config)#config=tf_config
            self.base_model = select_basemodel(self.seed, self.model_name, self.UNIREP_BATCH_SIZE, tf_config)
            self.sess.run(tf.global_variables_initializer())
    '''PtsRep序列映射部分'''

    def levenshtein_distance_matrix(self, a_list, b_list=None, verbose=False):
        """Computes an len(a_list) x len(b_list) levenshtein distance
        matrix.
        """
        if b_list is None:
            single_list = True
            b_list = a_list
        else:
            single_list = False
        
        H = np.zeros(shape=(len(a_list), len(b_list)))
        for i in range(len(a_list)):
            if verbose:
                print(i)
            
            if single_list:  
                # only compute upper triangle.
                for j in range(i+1,len(b_list)):
                    H[i,j] = Levenshtein.distance(a_list[i], b_list[j])
                    H[j,i] = H[i,j]
            else:
                for j in range(len(b_list)):
                    H[i,j] = Levenshtein.distance(a_list[i], b_list[j])

        return H

    def knr2ptsrep(self, knr_list):
        torch.set_num_threads(1)#限制CPU占用量
        device_num = 0
        torch.cuda.set_device(device_num)
        device = torch.device('cuda:%d' % device_num)
        if torch.cuda.is_available():
            print('MainDevice=', device)
        
        if self.model_name == "ePtsRep":
            pts_model_path = "/home/caiyi/seqrep/outputs/models/lm__21051402_evotune_out-in_resume/197_133649_model.pth.tar"#"/home/caiyi/seqrep/outputs/models/lm__21051402_evotune_out-in_resume/197_133649_model.pth.tar"
        if self.model_name == "PtsRep":
            pts_model_path = "/home/caiyi/seqrep/outputs/models/ptsrep__20121505_5_sota_right_n2_knnnet150_del_tor/50_model.pth.tar"
        if self.model_name == "Random_PtsRep":
            pts_model_path = "/share/caiyi/seqrep_outputs/models/lm__21051601_evotune_random/197_133649_model.pth.tar"
        ebd_list = []
        full_dataset = misc_utils.Knnonehot(knr_list)
        data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=self.UNIREP_BATCH_SIZE)#config['embed_batch_size']
        print("data_loader:", len(data_loader))

        with torch.no_grad():
            if self.pts_load_model_method == 'full_model':
                model = torch.load(pts_model_path, map_location=device)
            elif self.pts_load_model_method == 'state_dict':
                model = PtsRep(input_size=135, hidden_size=384, vocab_size=20, dropout=0.1).to('cuda:0')
                state_dict = torch.load(pts_model_path)
                model.load_state_dict(state_dict)
            else:
                raise NameError('No such model loading method!')
            model.eval()
            model.is_training = False

        for arrays in tqdm(data_loader, ascii=True):
            # print(arrays.shape)
            arrays = arrays.to(device).float()
            # pred = model(arrays[0]).float()
            pred = model(arrays).float()
            pred = pred.data.cpu().numpy()
            ebd_list.append(pred)
            # print(pred.shape)
        # ebd_reps = np.stack([np.mean(s, axis=0) for s in ebd_list], axis=0)
        if 'UniRep' in self.model_name or self.model_name == 'OneHot':
            ebd_reps = np.stack(ebd_list[0], 0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'lin':
            ebd_reps = np.vstack(ebd_list)
            ebd_reps = np.stack([np.mean(s, axis=0) for s in ebd_reps], axis=0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'nn':
            ebd_reps = np.vstack(ebd_list)
            # ebd_reps = np.stack(ebd_list[0], 0)
            ebd_reps = np.stack([np.mean(s, axis=0) for s in ebd_reps], axis=0)
        # ebd_reps = np.vstack(ebd_list)
        return ebd_reps
    
    def substitute(self, knr, ref_seq, seq):
        new_knr = np.copy(knr)
        muts = []
        for i, aa in enumerate(seq):
            if aa != ref_seq[i]:
                muts.append((i, aa))
        for pos, aa_name in muts:
            aa = self.AA_LIST.index(aa_name)
            # 200为KNR中位置编码使用的normalize
            struct_seq_len = knr.shape[0]
            idx = np.round(new_knr[:, :, 5] * 200) + np.arange(struct_seq_len).reshape(struct_seq_len, 1)
            tmp = new_knr[np.where(idx == pos)]
            tmp[:, -3:] = (self.hydropathicity[aa], self.bulkiness[aa], self.flexibility[aa])
            new_knr[np.where(idx == pos)] = tmp
        return new_knr

    def generate_pts_reps(self, seqs):
        pdb_av_GFP_path = "/share/joseph/seqtonpy/gfp/pdb/1emm_mod.pdb"
        pdb_sf_GFP_path = "/home/wangqihan/low-N-protein-engineering-master/泛化验证/pdb/sf_GFP/syn_neigh.pdb"
        ref_seq = seqs[0]
        if len(ref_seq) == 223:
            pdb_path = pdb_av_GFP_path
        elif len(ref_seq) == 231:
            pdb_path = pdb_sf_GFP_path
        # if Levenshtein.distance(self.reference_seq[self.seq_start:self.seq_end], ref_seq) >= 20:
        #     pdb_path = pdb_sf_GFP_path
        # elif Levenshtein.distance(self.reference_seq[self.seq_start:self.seq_end], ref_seq) < 20:
        #     pdb_path = pdb_av_GFP_path
        knr_list = []
        chain, model = 'A', '1'
        pdb_profile, atom_lines = process_pdb(pdb_path, atoms_type=['N', 'CA', 'C'])
        atoms_data = atom_lines[chain, model]
        coord_array_ca, struct_aa_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=['N', 'CA', 'C'])
        # print(len(coord_array), struct_aa_array, len(ref_seq))
        rep_utils.validate_aligning(struct_aa_array, ref_seq, allowed_mismatches=20)
        knr_ref = rep_utils.get_knn_150(coord_array, ref_seq)
        for seq in tqdm(seqs, ascii=True):
            # knr = rep_utils.get_knn_150(coord_array, seq)
            knr = self.substitute(knr_ref, ref_seq, seq)
            knr_list.append(knr)
        ebd_reps = self.knr2ptsrep(knr_list)
        # print(ebd_reps.shape)
        return ebd_reps
    
    '''UniRep映射部分'''

    def levenshtein_distance_matrix(self, a_list, b_list=None, verbose=False):
        """Computes an len(a_list) x len(b_list) levenshtein distance
        matrix.
        """
        if b_list is None:
            single_list = True
            b_list = a_list
        else:
            single_list = False
        
        H = np.zeros(shape=(len(a_list), len(b_list)))
        for i in range(len(a_list)):
            if verbose:
                print(i)
            
            if single_list:  
                # only compute upper triangle.
                for j in range(i+1,len(b_list)):
                    H[i,j] = Levenshtein.distance(a_list[i], b_list[j])
                    H[j,i] = H[i,j]
            else:
                for j in range(len(b_list)):
                    H[i,j] = Levenshtein.distance(a_list[i], b_list[j])

        return H
    
    def generate_uni_reps(self, seq_list):
        if 'babbler1900' == self.base_model.__class__.__name__:
            hiddens = [] 
            k = len(seq_list) // self.UNIREP_BATCH_SIZE
            if (len(seq_list) % self.UNIREP_BATCH_SIZE) > 0:
                k += 1
            for i in tqdm(range(k)):
                seq_list_k = seq_list[i*self.UNIREP_BATCH_SIZE : (i+1)*self.UNIREP_BATCH_SIZE]
                hidden_batch = self.base_model.get_all_hiddens(seq_list_k, self.sess)
                hiddens += hidden_batch
            if self.top_model_name == 'lin':
                rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            elif self.top_model_name == 'nn':
                rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
                # rep = np.stack(hiddens ,0)
            # assert len(seq_list) <= self.UNIREP_BATCH_SIZE
            # hiddens = self.base_model.get_all_hiddens(seq_list, self.sess)
            # rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            
        elif 'OneHotRegressionModel' == self.base_model.__class__.__name__:
            rep = self.base_model.encode_seqs(seq_list)
            
        return rep
    
    
    
    def get_fitness(self, seqs1, top_model):
        if "PtsRep" in self.model_name:
            seqs = []
            for seq in seqs1:
                seqs.append(seq[self.seq_start:self.seq_end])
            reps = self.generate_pts_reps(seqs)
        elif "UniRep" in self.model_name:
            seqs = seqs1
            reps = self.generate_uni_reps(seqs)
        else:
            raise NameError('Wrong value of `sampling_method`')
        yhat, yhat_std, yhat_mem = top_model.predict(reps, 
                return_std=True, return_member_predictions=True)
        if "PtsRep" in self.model_name:        
            nmut = self.levenshtein_distance_matrix(
                    [self.reference_seq[self.seq_start:self.seq_end]], list(seqs)).reshape(-1)
        elif "UniRep" in self.model_name:
            nmut = self.levenshtein_distance_matrix(
                    [self.reference_seq], list(seqs)).reshape(-1)
        mask = nmut > self.nmut_threshold
        yhat[mask] = -np.inf 
        yhat_std[mask] = 0 
        yhat_mem[mask,:] = -np.inf 
        return yhat, yhat_std, yhat_mem    

    '''seq_to_rep的方法'''

    def load_rep_by_seq(self, rep_path, seq_list):
        with open(self.seq2name_file) as f:
            lines = f.readlines()
        d = {}
        for line in lines:
            line = line.split()
            # print(line)
            d[line[0]] = line[1]
        rep_list = []
        if 'Alphafold' in self.model_name:
            for seq in seq_list:
                with open(f'{rep_path}/{d[seq]}.pkl', "rb") as f:
                    dick = pickle.load(f)
                    rep_list.append(dick["pair"])
        else:
            for seq in seq_list:
                rep_list.append(np.load(f'{rep_path}/{d[seq]}.npy'))
        if 'UniRep' in self.model_name or self.del_name == 'OneHot':
            rep_array = np.stack(rep_list, 0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'lin':
            rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'nn':
            rep_array = np.stack(rep_list, 0)
        elif 'Alphafold' in self.model_name and self.top_model_name == 'lin':
            rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
        elif 'Alphafold' in self.model_name and self.top_model_name == 'nn':
            rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
        return rep_array



    def load_rep_by_name(self, rep_path, name_list):
        rep_list = []
        if 'Alphafold' in self.model_name:
            for name in name_list:
                with open(f'{rep_path}/{name}.pkl', "rb") as f:
                    disk = pickle.load(f)
                    rep_list.append(disk['representations'][self.alph2_target])
        elif 'Openfold' in self.model_name:
            for name in name_list:
                with open(f'{rep_path}/{name}.pkl', "rb") as f:
                    disk = pickle.load(f)
                    rep_list.append(disk[self.openf_target])
        else:        
            for name in name_list:
                rep_list.append(np.load(f'{rep_path}/{name}.npy'))
        if 'UniRep' in self.model_name or self.model_name == 'OneHot':
            rep_array = np.stack(rep_list, 0)
            # rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'lin':
            rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
        elif 'PtsRep' in self.model_name and self.top_model_name == 'nn':
            rep_array = np.stack(rep_list, 0)
            # rep_array = np.stack([np.mean(s, axis=0) for s in rep_list],0)
        elif 'Alphafold' in self.model_name and self.top_model_name == 'lin':
            rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
        elif 'Alphafold' in self.model_name and self.top_model_name == 'nn':
            rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
        elif 'Openfold' in self.model_name and self.top_model_name == 'lin':
            rep_array = np.stack([np.mean(np.mean(s, axis=0), axis=0) for s in rep_list], 0)
        elif 'Openfold' in self.model_name and self.top_model_name == 'nn':
            rep_array = np.stack([np.mean(s, axis=0) for s in rep_list], 0)
        return rep_array

    def get_reps(self, sub_df, rep_src = 'generate', rep_path = None):
        if rep_src == 'load_by_seq':
            seqs = list(sub_df['seq'])
            reps = self.load_rep_by_seq(rep_path, seqs)
        elif rep_src == 'load_by_name':
            names = list(sub_df['name'])
            reps = self.load_rep_by_name(rep_path, names)
        elif rep_src == 'generate':
            seqs = list(sub_df['seq'])
            if 'PtsRep' in self.model_name:
                ref_seq = seqs[0]
                if Levenshtein.distance(self.reference_seq[self.seq_start:self.seq_end], ref_seq) >= 20:
                    seq_start = 1
                    seq_end = -6
                elif Levenshtein.distance(self.reference_seq[self.seq_start:self.seq_end], ref_seq) < 20:
                    seq_start = 6
                    seq_end = -9
                seqs_trimmed = []
                for seq in seqs:
                    seqs_trimmed.append(seq[seq_start:seq_end])#seqs_trimmed.append(seq[self.seq_start:self.seq_end])
                reps = self.generate_pts_reps(seqs_trimmed)
            elif 'UniRep' in self.model_name or 'onehot' in self.model_name:
                print('Setting up base model')
                reps = self.generate_uni_reps(seqs)
            elif "alphafold" in self.model_name:
                print('Setting up base model')

            else:
                raise NameError(f'Incorrect model name: {self.model_name}')
        return reps

    def get_train_reps(self, sub_train_df):
        train_reps = self.get_reps(sub_train_df, 
                                    self.train_rep_src, 
                                    self.train_rep_path, 
                                    )
        return train_reps
    
    def get_test_reps(self, test_df):
        test_reps = self.get_reps(test_df, 
                                    self.test_rep_src, 
                                    self.test_rep_path, 
                                    )
        return test_reps

    def predict_rep_fitness(self, top_model, reps):
        if self.top_model_name == 'nn':
            yhat = top_model.predict(reps)
        elif self.top_model_name == 'lin':
            yhat, yhat_std, yhat_mem = top_model.predict(reps, return_std=True, return_member_predictions=True)
        else :
            raise NameError(f'Incorrect model name: {self.model_name}')
        return yhat

    def create_dir_not_exist(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def save_result(self, pred_and_real, name_list, r, rho, r2, ndcg):
        if self.use_bright:
            if 'random' in self.sampling_method:
                output_file = f'bright_{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.n_train_seqs}_{self.training_objectives}_{self.seed}.p'
                output_data = np.array([pred_and_real, name_list], dtype=object)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/recall')
                np.save(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/recall/{output_file[:-2]}.npy', output_data)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/p_or_s')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/p_or_s/results_acti_pair_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                    f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\t{round(r2, 4)}\t{round(ndcg, 4)}\n')
                # output_file = f'bright_{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.n_train_seqs}_{self.training_objectives}_{self.seed}.p'
                # output_data = np.array([pred_and_real, name_list], dtype=object)
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}')
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}_{str(self.n_train_seqs)}')
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}_{str(self.n_train_seqs)}/recall')
                # np.save(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}_{str(self.n_train_seqs)}/recall/{output_file[:-2]}.npy', output_data)
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}_{str(self.n_train_seqs)}/p_or_s')
                # with open(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}_{str(self.n_train_seqs)}/p_or_s/results_acti_pair_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                #     f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\n')
            else:
                output_file = f'bright_{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.training_objectives}_{self.seed}.p'
                output_data = np.array([pred_and_real, name_list], dtype=object)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/bright_{self.model_name}_{self.sampling_method}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/bright_{self.model_name}_{self.sampling_method}/recall')
                np.save(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/bright_{self.model_name}_{self.sampling_method}/recall/{output_file[:-2]}.npy', output_data)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/bright_{self.model_name}_{self.sampling_method}/p_or_s')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/bright_{self.model_name}_{self.sampling_method}/p_or_s/results_acti_pair_{self.model_name}_{self.training_objectives}.txt', 'a') as f:
                    f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\t{round(r2, 4)}\t{round(ndcg, 4)}\n')
                # output_file = f'bright_{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.training_objectives}_{self.seed}.p'
                # output_data = np.array([pred_and_real, name_list], dtype=object)
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}')
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}')
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}/recall')
                # np.save(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}/recall/{output_file[:-2]}.npy', output_data)
                # self.create_dir_not_exist(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}/p_or_s')
                # with open(f'{self.output_dir}/{self.model_name}/bright_{self.training_objectives}_{self.sampling_method}/p_or_s/results_acti_pair_{self.model_name}_{self.training_objectives}.txt', 'a') as f:
                #     f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\n')
        else:
            if 'random' in self.sampling_method:
                output_file = f'{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.n_train_seqs}_{self.training_objectives}_{self.seed}.p'
                output_data = np.array([pred_and_real, name_list], dtype=object)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/recall')
                np.save(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/recall/{output_file[:-2]}.npy', output_data)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/p_or_s')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/p_or_s/results_acti_pair_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                    f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\t{round(r2, 4)}\t{round(ndcg, 4)}\n')
            else:
                output_file = f'{self.exp_name}_{self.model_name}_{self.sampling_method}_{self.training_objectives}_{self.seed}.p'
                # config['output_file'] = output_file
                # save_json_path = f'{self.output_dir}/configs/{output_file[:-2]}.json'
                # if not os.path.exists(save_json_path):
                #     shutil.copy(args.config_path, save_json_path)
                output_data = np.array([pred_and_real, name_list], dtype=object)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/{self.model_name}_{self.sampling_method}')
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/{self.model_name}_{self.sampling_method}/recall')
                np.save(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/{self.model_name}_{self.sampling_method}/recall/{output_file[:-2]}.npy', output_data)
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/{self.model_name}_{self.sampling_method}/p_or_s')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}/{self.model_name}_{self.sampling_method}/p_or_s/results_acti_pair_{self.model_name}_{self.training_objectives}.txt', 'a') as f:
                    f.write(f'{output_file[:-2]}\t{str(self.seed)}\t{round(r, 4)}\t{round(rho, 4)}\t{round(r2, 4)}\t{round(ndcg, 4)}\n')

    def save_train_set_proportion(self, sub_train_df):
        if self.use_bright:
            if self.sampling_method == "random_1-3":
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/proportion')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/proportion/proportion_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                        f.write(f'{str(self.seed)}\t3_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 3]["quantitative_function"]))}\t2_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 2]["quantitative_function"]))}\t1_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 1]["quantitative_function"]))}\n')
            elif self.sampling_method == "random_1-2":
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/proportion')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/bright_{self.model_name}_{self.sampling_method}/proportion/proportion_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                        f.write(f'{str(self.seed)}\t2_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 2]["quantitative_function"]))}\t1_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 1]["quantitative_function"]))}\n')
            else:
                raise NameError(f'Incorrect model name: {self.sampling_method}')
        else:
            if self.sampling_method == "random_1-3":
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/proportion')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/proportion/proportion_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                        f.write(f'{str(self.seed)}\t3_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 3]["quantitative_function"]))}\t2_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 2]["quantitative_function"]))}\t1_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 1]["quantitative_function"]))}\n')
            elif self.sampling_method == "random_1-2":
                self.create_dir_not_exist(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/proportion')
                with open(f'{self.output_dir}/{self.top_model_name}/{self.training_objectives}_{str(self.n_train_seqs)}/{self.model_name}_{self.sampling_method}/proportion/proportion_{self.model_name}_{self.n_train_seqs}_{self.training_objectives}.txt', 'a') as f:
                        f.write(f'{str(self.seed)}\t2_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 2]["quantitative_function"]))}\t1_mutation_proportion\t{len(list(sub_train_df[sub_train_df["distance"] == 1]["quantitative_function"]))}\n')
            else:
                raise NameError(f'Incorrect model name: {self.sampling_method}')
        
    def mcmc_design(self, top_model, sub_train_df, train_reps):
        print("design_method:" ,self.design_method)
        if self.design_method == "MCMC":
            sa_results = A006_common.anneal(
                    self.init_seqs, 
                    k=self.SIM_ANNEAL_K, 
                    T_max=self.T_max, 
                    mu_muts_per_seq=self.mu_muts_per_seq,
                    top_model = top_model,
                    get_fitness_fn=self.get_fitness,
                    n_iter=self.sa_n_iter, 
                    decay_rate=self.temp_decay_rate,
                    min_mut_pos=self.min_mut_pos,
                    max_mut_pos=self.max_mut_pos)
        elif self.design_method == 'GA':
            sa_results = ga.anneal(
                    self.init_seqs, 
                    k=self.SIM_ANNEAL_K, 
                    T_max=self.T_max, 
                    mu_muts_per_seq=self.mu_muts_per_seq,
                    top_model = top_model,
                    get_fitness_fn=self.get_fitness,
                    n_iter=self.sa_n_iter, 
                    decay_rate=self.temp_decay_rate,
                    min_mut_pos=self.min_mut_pos,
                    max_mut_pos=self.max_mut_pos)
        results = {
                    'sa_results': sa_results,
                    'top_model': top_model,
                    'train_df': sub_train_df,
                    'train_seq_reps': train_reps,
                    'base_model': self.model_name
                }

        return results

    def save_design_datas(self, results):
        if self.use_bright:
            if self.sampling_method == "random":
                with open(f'{self.output_dir}/design/{self.exp_name}_{self.training_objectives}_{self.model_name}_bright_{self.sampling_method}_{self.n_train_seqs}_{self.design_method}_{self.seed}.p', 'wb') as f:
                    pickle.dump(file=f, obj=results)
            else:
                with open(f'{self.output_dir}/design/{self.exp_name}_{self.training_objectives}_{self.model_name}_brighr_{self.sampling_method}_{self.design_method}_{self.seed}.p', 'wb') as f:
                    pickle.dump(file=f, obj=results)
        else:
            if self.sampling_method == "random":
                with open(f'{self.output_dir}/design/{self.exp_name}_{self.training_objectives}_{self.model_name}_{self.sampling_method}_{self.n_train_seqs}_{self.design_method}_{self.seed}.p', 'wb') as f:
                    pickle.dump(file=f, obj=results)
            else:
                with open(f'{self.output_dir}/design/{self.exp_name}_{self.training_objectives}_{self.model_name}_{self.sampling_method}_{self.design_method}_{self.seed}.p', 'wb') as f:
                    pickle.dump(file=f, obj=results)

