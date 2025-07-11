from tqdm import tqdm
import numpy as np
import torch
import sys
from low_n_utils import paths
from low_n_utils import models
from torch.utils.data import DataLoader
from low_n_utils.unirep import babbler1900 as babbler
from low_n_utils.process_pdb import process_pdb
from low_n_utils import rep_utils
import Levenshtein
import misc_utils
import modules
from misc_utils import PtsRep
from ..import config

class SeqEbdPtsRep(modules.ParameterInitialization):

    def __init__(self, config, device, top_model = None, reference_seq = None, nmut_threshold = 15):
        super(SeqEbdPtsRep, self).__init__(config)
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
        self.device = device
        self.top_model = top_model
        self.reference_seq = reference_seq
        self.nmut_threshold = nmut_threshold
    
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
        ebd_list = []
        full_dataset = misc_utils.Knnonehot(knr_list)
        data_loader = DataLoader(dataset=full_dataset, shuffle=False, batch_size=3500)#config['embed_batch_size']
        print("data_loader:", len(data_loader))

        with torch.no_grad():
            if config.PARAMETER['load_model_method'] == 'full_model':
                model = torch.load(self.model_path, map_location=self.device)
            elif config.PARAMETER['load_model_method'] == 'state_dict':
                model = PtsRep(input_size=135, hidden_size=384, vocab_size=20, dropout=0.1).to('cuda:0')
                state_dict = torch.load(self.model_path)
                model.load_state_dict(state_dict)
            else:
                raise NameError('No such model loading method!')
            model.eval()
            model.is_training = False

        for arrays in tqdm(data_loader, ascii=True):
            print(arrays.shape)
            arrays = arrays.to(self.device).float()
            # pred = model(arrays[0]).float()
            pred = model(arrays).float()
            pred = pred.data.cpu().numpy()
            ebd_list.append(pred)
            print(pred.shape)
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

    def generate_reps(self, seqs):
        knr_list = []
        chain, model = 'A', '1'
        pdb_profile, atom_lines = process_pdb(self.pdb_path, atoms_type=['N', 'CA', 'C'])
        atoms_data = atom_lines[chain, model]
        coord_array_ca, struct_aa_array, coord_array = rep_utils.extract_coord(atoms_data, atoms_type=['N', 'CA', 'C'])
        print(len(coord_array))
        ref_seq = seqs[0]
        rep_utils.validate_aligning(struct_aa_array, ref_seq, allowed_mismatches=20)
        knr_ref = rep_utils.get_knn_150(coord_array, ref_seq)
        for seq in tqdm(seqs, ascii=True):
            # knr = rep_utils.get_knn_150(coord_array, seq)
            knr = self.substitute(knr_ref, ref_seq, seq)
            knr_list.append(knr)
        ebd_reps = self.knr2ptsrep(knr_list)
        print(ebd_reps.shape)
        return ebd_reps
    
    def get_fitness(self, seqs1):
        seqs = []
        for seq in seqs1:
            seqs.append(seq[self.seq_start:self.seq_end])
        reps = self.generate_reps(seqs)
        yhat, yhat_std, yhat_mem = self.top_model.predict(reps, 
                return_std=True, return_member_predictions=True)
                
        nmut = self.levenshtein_distance_matrix(
                [self.reference_seq[self.seq_start:self.seq_end]], list(seqs)).reshape(-1)
        
        mask = nmut > self.nmut_threshold
        yhat[mask] = -np.inf 
        yhat_std[mask] = 0 
        yhat_mem[mask,:] = -np.inf 
        return yhat, yhat_std, yhat_mem

class SeqEbdUniRep(modules.ParameterInitialization):

    def __init__(self, sess, base_model, top_model = None, reference_seq = None, nmut_threshold = 15):
        super(SeqEbdUniRep, self).__init__(config)
        self.base_model = base_model
        self.sess = sess
        self.top_model = top_model
        self.reference_seq = reference_seq
        self.nmut_threshold = nmut_threshold
    
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
    
    def generate_reps(self, seq_list):
        if 'babbler1900' == self.base_model.__class__.__name__:
            assert len(seq_list) <= self.UNIREP_BATCH_SIZE
            hiddens = self.base_model.get_all_hiddens(seq_list, self.sess)
            rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            
        elif 'OneHotRegressionModel' == self.base_model.__class__.__name__:
            rep = self.base_model.encode_seqs(seq_list)
            
        return rep
    
    def get_fitness(self, seqs):
        reps = self.generate_reps(seqs)
        print("ebd_size:", reps.shape)
        yhat, yhat_std, yhat_mem = self.top_model.predict(reps, 
                return_std=True, return_member_predictions=True)
                
        nmut = self.levenshtein_distance_matrix(
                [self.reference_seq], list(seqs)).reshape(-1)
        # print(nmut)
        
        mask = nmut > self.nmut_threshold
        yhat[mask] = -np.inf 
        yhat_std[mask] = 0 
        yhat_mem[mask,:] = -np.inf 
        return yhat, yhat_std, yhat_mem


def select_basemodel(model_name, UNIREP_BATCH_SIZE, tf_config, sess):
    if model_name == 'UniRep':
        UNIREP_WEIGHT_PATH = '/home/caiyi/github/unirep_embedding/1900_weights/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config=tf_config, sess=sess)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    elif model_name == 'eUniRep':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH, config=tf_config, sess=sess)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_1_WEIGHT_PATH)
    elif model_name == 'eunirep_2':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH, config=tf_config, sess=sess)
        print('Loading weights from:', paths.GFP_ET_GLOBAL_INIT_2_WEIGHT_PATH)
    elif model_name == 'Random_UniRep':
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH, config=tf_config, sess=sess)
        print('Loading weights from:', paths.GFP_ET_RANDOM_INIT_1_WEIGHT_PATH)
    elif model_name =='onehot':
        # Just need it to generate one-hot reps.
        # Top model created within OneHotRegressionModel doesn't actually get used.
        base_model = models.OneHotRegressionModel('EnsembledRidge')
    elif model_name == 'eUniRep_petase': 
        UNIREP_WEIGHT_PATH = '/home/caiyi/unirep/eunirep_petase_21091902_30/'
        base_model = babbler(batch_size=UNIREP_BATCH_SIZE, model_path=UNIREP_WEIGHT_PATH, config=tf_config, sess=sess)
        print('Loading weights from:', UNIREP_WEIGHT_PATH)
    else:
        assert False, 'Unsupported base model'
    return base_model

# UNIREP_BATCH_SIZE = cd.PARAMETER['embed_batch_size']
        # if 'babbler1900' == self.base_model.__class__.__name__:
        #     assert len(seq_list) <= self.UNIREP_BATCH_SIZE
        #     hiddens = [] 
        #     k = len(seq_list) // 1000
        #     if (len(seq_list) % 1000) > 0:
        #         k += 1
        #     for i in tqdm(range(k)):
        #         seq_list_k = seq_list[i*1000 : (i+1)*1000]
        #         print(len(seq_list_k))
        #         hidden_batch = self.base_model.get_all_hiddens(seq_list_k, self.sess)
        #         hiddens += hidden_batch
        #     # hiddens = base_model.get_all_hiddens(seq_list, sess)
        #     rep = np.stack([np.mean(s, axis=0) for s in hiddens],0)
            
        # elif 'OneHotRegressionModel' == self.base_model.__class__.__name__:
        #     rep = self.base_model.encode_seqs(seq_list)

