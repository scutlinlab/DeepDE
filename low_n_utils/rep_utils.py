# -*- coding: utf-8 -*
import numpy as np
import os
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
# import matplotlib.colors as mplcolors
# import matplotlib.pyplot as plt

from math import ceil
from typing import Dict, Union, Tuple, List


'''
FROM geo.py
'''

def rotation(vec, axis, theta_):
    """[罗德里格旋转公式]

    Arguments:
        vec {[npy]} -- [原始坐标 [[x1,y1,z1],[x2,y2,z2]...]]
        theta {[float]} -- [转角 弧度值]
        axis {[npy]} -- [转轴 [x,y,z]]

    Returns:
        [npy] -- [旋转所得坐标 [[x1,y1,z1],[x2,y2,z2]...]]
    """
    theta = theta_ + np.pi
    cos = np.cos(theta)
    vec_rot = cos*vec + \
        np.sin(theta)*np.cross(axis, vec) + \
        (1 - cos) * batch_dot(vec, axis).reshape(-1, 1) * axis
    if len(vec_rot) == 1:
        vec_rot = vec_rot[0]
    return vec_rot


def batch_dot(vecs1, vecs2):
    if len(vecs2.shape) == 1:
        vecs2 = vecs2.reshape(1, -1)
    return np.diag(np.matmul(vecs1, vecs2.T))


def get_torsion(vec1, vec2, axis):
    """[计算以axis为轴，向量1到向量2的旋转角]

    Arguments:
        vec1 {[npy]} -- [向量1 [x,y,z]]
        vec2 {[npy]} -- [向量2 [x,y,z]]
        axis {[npy]} -- [转轴axis [x,y,z]]

    Returns:
        [float] -- [旋转角]
    """
    n = np.cross(axis, vec2)
    n2 = np.cross(vec1, axis)
    sign = np.sign(batch_cos(vec1, n))
    angle = np.arccos(batch_cos(n2, n))
    torsion = sign*angle
    if len(torsion) == 1:
        return torsion[0]
    else:
        return torsion


def get_len(vec):
    return np.linalg.norm(vec, axis=-1)


def norm(vec):
    return vec / get_len(vec).reshape(-1, 1)


def get_angle(vec1, vec2):
    return np.arccos(np.dot(norm(vec1), norm(vec2)))


def batch_cos(vecs1, vecs2):
    cos = np.diag(np.matmul(norm(vecs1), norm(vecs2).T))
    cos = np.clip(cos, -1, 1)
    return cos


AA_ALPHABET = {'A': 'ALA', 'F': 'PHE', 'C': 'CYS', 'D': 'ASP', 'N': 'ASN',
               'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU',
               'I': 'ILE', 'K': 'LYS', 'M': 'MET', 'P': 'PRO', 'R': 'ARG',
               'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA_ALPHABET_REV = {'ALA': 'A', 'PHE': 'F', 'CYS': 'C', 'ASP': 'D', 'ASN': 'N',
              	   'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L',
              	   'ILE': 'I', 'LYS': 'K', 'MET': 'M', 'PRO': 'P', 'ARG': 'R',
              	   'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AA_NUM = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4,
          'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9,
          'I': 10, 'K': 11, 'M': 12, 'P': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
AA_HYDROPATHICITY_INDEX = {'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5,
                           'E': -3.5, 'H': -3.2, 'P': -1.6, 'Y': -1.3, 'W': -0.9,
                           'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9,
                           'C': 2.5, 'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5}
AA_BULKINESS_INDEX = {'R': 14.28, 'K': 15.71, 'N': 12.82, 'D': 11.68, 'Q': 14.45,
                      'E': 13.57, 'H': 13.69,  'P': 17.43, 'Y': 18.03, 'W': 21.67,
                      'S': 9.47, 'T': 15.77, 'G': 3.4, 'A': 11.5, 'M': 16.25,
                      'C': 13.46, 'F': 19.8, 'L': 21.4, 'V': 21.57, 'I': 21.4}
AA_FLEXIBILITY_INDEX = {'R': 2.6, 'K': 1.9, 'N': 14., 'D': 12., 'Q': 4.8,
                        'E': 5.4, 'H': 4., 'P': 0.05, 'Y': 0.05, 'W': 0.05,
                        'S': 19., 'T': 9.3, 'G': 23., 'A': 14., 'M': 0.05,
                        'C': 0.05, 'F': 7.5, 'L': 5.1, 'V': 2.6, 'I': 1.6}
AA_PROPERTY = {}

# CAUTIOUS: MAY HAVE COMPATIBILITY PROBLEMS
for aa in AA_HYDROPATHICITY_INDEX.keys():
    AA_PROPERTY.update({aa: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                             AA_BULKINESS_INDEX[aa] / 21.67,
                             (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})
    aa_long = AA_ALPHABET[aa]
    AA_PROPERTY.update({aa_long: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                              AA_BULKINESS_INDEX[aa] / 21.67,
                              (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})


def MapDis(coo):
    return squareform(pdist(coo, metric='euclidean')).astype('float32')


'''
FROM knn_utils.py
'''
def extract_coord(atoms_data, atoms_type):
    coord_array_ca = np.zeros((ceil(len(atoms_data) / len(atoms_type)), 3))  # CA坐标, shape: L * 3
    coord_array_all = np.zeros((len(atoms_data), 3))  # 所有backbone原子(或atom_types中的原子)坐标, shape: 4L * 3
    aa_names = []
    for i in range(len(atoms_data)):
        coord_array_all[i] = [float(atoms_data[i][j]) for j in range(6, 9)] 
        # 写法可能不合适,未考虑氨基酸内部原子顺序不一致的情况
        if i % len(atoms_type) == atoms_type.index('CA'):
            coord_array_ca[i // len(atoms_type)] = [float(atoms_data[i][j]) for j in range(6, 9)] 
            aa_names.append(atoms_data[i][3][-3::])
    aa_names_array = np.array(aa_names)  # shape: L * 1
    return coord_array_ca, aa_names_array, coord_array_all


def read_fasta(file_path):
    seq_dict: Dict[str, str] = {}  # {seq_name -> fasta_seq}
    seq_file = open(file_path, 'r')
    seq_data = seq_file.readlines()
    # 读取fasta文件

    for i in range(len(seq_data)):
        if seq_data[i][0] == '>':
            seq_name = seq_data[i][1:-1]
            seq_dict[seq_name] = ''
            j = 1 
            while True:
                if i + j >= len(seq_data) or seq_data[i + j][0] == '>':
                    break
                else:
                    seq_dict[seq_name] += ''.join(seq_data[i + j].split())
                j += 1
    return seq_dict


def seq2array(aa_seq: str) -> np.ndarray:
    aa_seq = list(aa_seq)
    for i in range(len(aa_seq)):
        aa_seq[i] = AA_ALPHABET[aa_seq[i]]
    aa_seq_array = np.array(aa_seq)
    return aa_seq_array


def validate_aligning(strcut_seq, substitute_seq, allowed_mismatches=50):
    
    assert len(strcut_seq) == len(substitute_seq)
    mismatches = 0
    for i, aa in enumerate(strcut_seq):
        if substitute_seq[i] != AA_ALPHABET_REV[aa]:
            mismatches += 1
    print('mismatches between structure seq and substitute seq:', mismatches)
    assert mismatches <= allowed_mismatches, 'Structure and substitute seq not aligned!'


def compare_len(coord_array, aa_array, atoms_type):
    atoms = len(atoms_type)
    # print(coord_array.shape[0] / atoms, aa_array.shape[0])
    if len(coord_array) / atoms > len(aa_array):
        raise("Seq too short!")
    elif len(coord_array) / atoms < len(aa_array):
        raise("Seq too long!")

'''
END OF knn_utils.py
'''


# 150: 15*9, (spherical_coord*4 + distance*1 + seq_relative_pos*1 + aa_property*3)
def get_knn_150(coo, aa):
    compare_len(coo, aa, ['N', 'CA', 'C'])
    ca_coo = []
    for i in range(len(coo)):
        if i % 3 == 1:
            ca_coo.append(coo[i])
    ca_coo = np.array(ca_coo)
    knn_spher = StrucRep('knn', 'property', 200).knn_struc_rep(ca_coo, aa, k=15)
    # return get_knn_150_append((knn_spher, coo), aa, 30)
    return knn_spher


class StrucRep(object):
    def __init__(self, struc_format='knn', aa_format='property', index_norm=200):
        self.aa_encoder = AminoacidEncoder(aa_format)
        self.index_norm = index_norm
        if struc_format == 'knn':
            self.struc_rep = self.knn_struc_rep
        elif struc_format == 'image':
            self.struc_rep = self.image_struc_rep
        elif struc_format == 'conmap':
            self.struc_rep = self.contact_map
        elif struc_format == 'dismap':
            self.struc_rep = self.distance_map

    def knn_struc_rep(self, ca, seq, k=15):
        dismap = MapDis(ca)
        nn_indexs = np.argsort(dismap, axis=1)[:, :k]
        # nn_indexs = dismap[:, :k]
        # print(nn_indexs)
        relative_indexs = nn_indexs.reshape(-1, k, 1) - \
            nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
        relative_indexs /= self.index_norm
        seq_embeded = self.aa_encoder.encode(seq)
        knn_feature = np.array(seq_embeded)[nn_indexs]
        knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
        knn_distance = np.array(knn_distance).reshape(-1, k, 1)

        tgt_x = np.array([0, 1, 0])
        rot_axis_y = tgt_x
        tgt_y = np.array([1, 0, 0])
        ori_x = norm(ca[1:] - ca[:-1])
        ori_y = np.concatenate((ori_x[1:], -(ori_x[np.newaxis, -2])))
        ori_x = np.concatenate((ori_x, ori_x[np.newaxis, -1]))
        ori_y = np.concatenate((ori_y, ori_y[np.newaxis, -1]))

        rot_axis_x = norm(np.cross(ori_x, tgt_x))
        tor_x = get_torsion(ori_x, tgt_x, rot_axis_x)
        ori_y_rot = rotation(ori_y, rot_axis_x, tor_x.reshape(-1, 1))
        ori_y_proj = ori_y_rot.copy()
        ori_y_proj[:, 1] = 0.
        ori_y_proj = norm(ori_y_proj)
        l_ori_y_proj = len(ori_y_proj)
        tor_y = get_torsion(ori_y_proj,
                                np.tile(tgt_y, (l_ori_y_proj, 1)),
                                np.tile(rot_axis_y, (l_ori_y_proj, 1)))

        knn_sincos = []
        for i, center in enumerate(ca):
            ca_ = ca - center
            global_indexs = nn_indexs[i]
            ca_xrot = rotation(ca_[global_indexs],
                                   np.tile(rot_axis_x[i], (k, 1)),
                                   np.tile(tor_x[i], (k, 1)))
            ca_rot = rotation(ca_xrot,
                                  np.tile(rot_axis_y, (k, 1)),
                                  np.tile(tor_y[i], (k, 1)))

            sin_1 = ca_rot[1:, 0] / \
                np.sqrt(np.square(ca_rot[1:, 0]) + np.square(ca_rot[1:, 1]))
            cos_1 = ca_rot[1:, 1] / \
                np.sqrt(np.square(ca_rot[1:, 0]) + np.square(ca_rot[1:, 1]))

            cos_2 = ca_rot[1:, 2]/knn_distance[i, 1:].reshape(-1)
            sin_2 = np.sqrt(1-np.square(cos_2))

            knn_sincos.append(np.concatenate([np.zeros((1, 4)), np.array(
                [sin_1, cos_1, sin_2, cos_2]).T]))

        knn_sincos = np.array(knn_sincos)
        knn_rep = np.concatenate(
            (knn_sincos, knn_distance, relative_indexs, knn_feature), -1)
        return knn_rep.astype('float32')

    def contact_map(self, ca, seq='', cutoff=8):
        dismap = MapDis(ca)
        conmap = np.zeros_like(dismap)
        conmap[dismap < cutoff] = 1.
        return conmap.astype('float32')

    def distance_map(self, ca, seq=''):
        return MapDis(ca).astype('float32')


class AminoacidEncoder(object):
    def __init__(self, aa_format='property'):
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.index = {}
        for aa in self.aa_list:
            self.index[aa] = self.aa_list.index(aa)

        if aa_format == 'onehot':
            self.encoder = np.eye(20)

        elif aa_format == 'property':
            self.hydropathicity = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
            self.bulkiness = [11.5, 13.46, 11.68, 13.57, 19.8, 3.4, 13.69, 21.4, 15.71, 21.4,
                              16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03]
            self.flexibility = [14.0, 0.05, 12.0, 5.4, 7.5, 23.0, 4.0, 1.6, 1.9, 5.1,
                                0.05, 14.0, 0.05, 4.8, 2.6, 19.0, 9.3, 2.6, 0.05, 0.05]
            self.property_norm()

            self.encoder = np.stack([self.hydropathicity,
                                     self.bulkiness,
                                     self.flexibility]).T.astype('float32')

    def property_norm(self):
        self.hydropathicity = (5.5 - np.array(self.hydropathicity)) / 10
        self.bulkiness = np.array(self.bulkiness) / max(self.bulkiness)
        self.flexibility = (25 - np.array(self.flexibility)) / 25

    def encode(self, seq):
        if len(seq[0]) == 3:
            seq = [AA_ALPHABET_REV[aa] for aa in seq]
        indexs = np.array([self.index[aa] for aa in seq])
        return self.encoder[indexs]
