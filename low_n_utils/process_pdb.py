from typing import Dict, List, Tuple
import os

# 以原子在PDB中实际的排列顺序排列
# atoms_type = ['N', 'CA', 'C', 'O']


# 尚未实现的功能：残基序号跳跃但不断链的情况识别， 删除missing atom的aa，将非天然氨基酸插入到atom_lines
# noinspection PyShadowingNames
def process_pdb(file_name, atoms_type=['CA']):
    '''
    len_seq: chain -> len_seq; len_struct_seq: (chain, model) -> len_struct_seq
    repeated_atoms: (chain, model) -> count_repeat_atoms; missing_aa_pos: (chain, model) -> {position -> count}
    residue_num_skip: (chain, model) -> {position -> count}; aa_missing_atom_pos: (chain, model) -> [position]
    extra_aa_pos: (chain, model) -> {position -> count}
    mod_ori_aa_name: [(modified_aa_name, original_aa_name)]; mod_aa_pos: chain -> {position -> mod_aa_name}
    '''

    pdb_profile = {'num_models': 0, 'num_chains': 0, 'name_chains': [], 'missing_aa_p': False, 'missing_aa_pos': {},
                   'extra_aa_p': False, 'extra_aa_pos': {}, 'start_with_minus_p': False, 'adhered_data_p': False,
                   'aa_missing_atom_p': False, 'aa_missing_atom_pos': {}, 'repeated_atoms': {}, 'residue_num_skip': {},
                   'len_seq': {}, 'len_struct_seq': {}, 'struct_seq_span': {}, 'struct_seq_ends': {},
                   'mod_aa_p': False, 'mod_ori_aa_name': [], 'mod_aa_pos': {}}
    atom_lines: Dict[Tuple[str, str], List[List[str]]] = {}  # (chain, model) -> atom_lines
    name_chains: List[str] = []
    missing_aa_pos: Dict[Tuple[str, str], Dict[int, int]] = {}
    extra_aa_pos: Dict[Tuple[str, str], Dict[int, int]] = {}
    aa_missing_atom_pos: Dict[Tuple[str, str], List[str]] = {}
    repeated_atoms: Dict[Tuple[str, str], int] = {}
    len_seq: Dict[str, int] = {}
    len_struct_seq: Dict[Tuple[str, str], int] = {}
    struct_seq_span: Dict[Tuple[str, str], int] = {}
    struct_seq_ends: Dict[Tuple[str, str], Tuple[int, int]] = {}
    mod_ori_aa_name: List[Tuple[str, str]] = []
    mod_aa_pos: Dict[str, Dict[str, str]] = {}

    file = open((file_name), 'r')
    pdb_lines = file.readlines()
    curr_model = '1'

    # 读取PDB文件，获取所需信息
    for line in pdb_lines:
        line_split = line.split()

        # 获得序列长度
        # noinspection SpellCheckingInspection
        if not line_split:
            continue
        if line_split[0] == 'SEQRES' and line_split[1] == '1':
            len_seq[line_split[2]] = int(line_split[3])

        # 获得模型数量
        if line_split[0] == 'MODEL':
            curr_model = line_split[1]
            pdb_profile['num_models'] += 1

        # 获取非天然氨基酸信息
        # noinspection SpellCheckingInspection
        if line_split[0] == 'MODRES':
            pdb_profile['mod_aa_p'] = True
            if (line_split[2], line_split[5]) not in mod_ori_aa_name:
                mod_ori_aa_name.append((line_split[2], line_split[5]))
            try:
                mod_aa_pos[line_split[3]][line_split[4]] = line_split[2]
            except KeyError:
                mod_aa_pos[line_split[3]] = {line_split[4]: line_split[2]}

        # 读取所需原子信息的行
        if line_split[0] == 'ATOM' and line_split[2] in atoms_type:
            # 如果chain长度大于1，说明chain字段与residue_num字段黏连
            if len(line_split[4]) > 1:
                pdb_profile['adhered_data_p'] = True
                line_split.insert(4, line_split[4][0])
                line_split[5] = line_split[5][1:]
            try:
                atom_lines[line_split[4], curr_model].append(line_split)
            except KeyError:
                atom_lines[line_split[4], curr_model] = [line_split]

    # 统计每个链和模型中的结构序列长度、缺失氨基酸的位置、氨基酸缺失原子的位置、重复原子数量、残基序号跳跃位置等情况
    for chain, model in atom_lines:
        aa_atom_count = {}   # 各氨基酸中原子的数目(所选原子) residue_num -> atom_count
        repeated_atoms[chain, model] = 0
        missing_aa_pos[chain, model] = {}
        extra_aa_pos[chain, model] = {}
        aa_missing_atom_pos[chain, model] = []
        atoms_to_delete = []

        # 获得链的名称列表
        if chain not in name_chains:
            name_chains.append(chain)
        curr_chain_lines = atom_lines[chain, model]

        # 处理该链和模型的每一行
        for i in range(len(curr_chain_lines)):
            # 标记重复出现的原子
            repeated_atom_p = 0
            # 当发现同一氨基酸中重复出现的原子
            if curr_chain_lines[i - 1][2] == curr_chain_lines[i][2] and \
                    curr_chain_lines[i - 1][5] == curr_chain_lines[i][5]:
                repeated_atom_p = 1
                # 标记重复出现的原子中概率小的原子，待删除
                if curr_chain_lines[i - 1][-3] <= curr_chain_lines[i][-3]:
                    atoms_to_delete.append(i - 1)
                else:
                    atoms_to_delete.append(i)
                repeated_atoms[chain, model] += 1

            # 得到每个氨基酸包含的原子数，以检查原子的缺失或冗余
            try:
                if repeated_atom_p == 0:
                    aa_atom_count[curr_chain_lines[i][5]] += 1
            except KeyError:
                aa_atom_count[curr_chain_lines[i][5]] = 1

        # 删除重复出现的原子中概率小的原子(由于curr_chain_lines使用字典索引赋值，可同步删除atom_lines中内容)
        for i in atoms_to_delete[::-1]:
            del curr_chain_lines[i]

        # 分析每个氨基酸中包含原子个数和氨基酸序号
        i = 0  # 因为与下一行比较，因此只比较到倒数第二行，i用于计数
        extra_aa_count = 0  # 连续的额外氨基酸的计数
        last_extra_aa = 0  # 上一个额外氨基酸的编号(仅数字部分)
        for aa in aa_atom_count:
            # 处理额外的(编号为数字+字母)的氨基酸
            try:
                int(aa)
            # 当发现额外的氨基酸
            except ValueError:
                if int(aa[:-1]) == last_extra_aa:
                    extra_aa_count += 1
                else:
                    extra_aa_count = 1
                aa = int(aa[:-1]) + 0.01 * extra_aa_count
                extra_aa_pos[chain, model][int(aa)] = extra_aa_count
                pdb_profile['extra_aa_p'] = True
                last_extra_aa = int(aa)

            # 当发现缺失氨基酸
            if str(int(aa) + 1) not in aa_atom_count and i < len(aa_atom_count) - 1:
                # 计算连续缺失氨基酸的数量
                missing_aa_count = 1
                while True:
                    if str(int(aa) + missing_aa_count + 1) in aa_atom_count:
                        break
                    else:
                        missing_aa_count += 1
                # 保存此处缺失氨基酸的位置和数量
                missing_aa_pos[chain, model][int(aa) + 1] = missing_aa_count
                pdb_profile['missing_aa_p'] = True
            # 当发现某氨基酸缺失原子
            if aa_atom_count[str(int(aa))] < len(atoms_type):
                aa_missing_atom_pos[chain, model].append(aa)
                pdb_profile['aa_missing_atom_p'] = True
            i += 1

        # 删除缺失原子的氨基酸
        for aa in aa_missing_atom_pos[chain, model]:
            del aa_atom_count[aa]
        atoms_to_delete = []
        for i in range(len(curr_chain_lines)):
            if curr_chain_lines[i][5] in aa_missing_atom_pos[chain, model]:
                atoms_to_delete.append(i)
        for atom in atoms_to_delete[::-1]:
            del curr_chain_lines[atom]

        # 结构序列的长度，起始点等统计
        aa_nums = list(aa_atom_count.keys())
        aa_num_start = int(aa_nums[0])
        aa_num_end = int(aa_nums[-1])
        struct_seq_ends[chain, model] = (aa_num_start, aa_num_end)
        struct_seq_span[chain, model] = aa_num_end - aa_num_start + 1
        len_struct_seq[chain, model] = len(aa_atom_count)

        if not missing_aa_pos[chain, model]:
            del missing_aa_pos[chain, model]
        if not extra_aa_pos[chain, model]:
            del extra_aa_pos[chain, model]
        if not aa_missing_atom_pos[chain, model]:
            del aa_missing_atom_pos[chain, model]
        if not repeated_atoms[chain, model]:
            del repeated_atoms[chain, model]

    if pdb_profile['num_models'] == 0:
        pdb_profile['num_models'] = 1
    pdb_profile['num_chains'] = len(name_chains)
    pdb_profile['name_chains'] = name_chains
    pdb_profile['len_seq'] = len_seq
    pdb_profile['len_struct_seq'] = len_struct_seq
    pdb_profile['missing_aa_pos'] = missing_aa_pos
    pdb_profile['extra_aa_pos'] = extra_aa_pos
    pdb_profile['aa_missing_atom_pos'] = aa_missing_atom_pos
    pdb_profile['repeated_atoms'] = repeated_atoms
    pdb_profile['struct_seq_ends'] = struct_seq_ends
    pdb_profile['struct_seq_span'] = struct_seq_span
    pdb_profile['mod_aa_pos'] = mod_aa_pos
    pdb_profile['mod_ori_aa_name'] = mod_ori_aa_name
    for chain, model in struct_seq_ends:
        if int(struct_seq_ends[chain, model][0]) < 0:
            pdb_profile['start_with_minus_p'] = True

    # 检查处理后的结果
    # for chain, model in atom_lines:
    #     for i in range(len(atom_lines[chain, model])):
    #         print(atom_lines[chain, model][i])
    #         pass
    # print(atom_lines)
    print(file_name, pdb_profile)
    print(len(atom_lines), atom_lines.keys())
    return pdb_profile, atom_lines


if __name__ == '__main__':
    for pdb_file_name in os.listdir(pdb_path):
        pdb_profile, processed_atom_lines = process_pdb(pdb_file_name)
        print(pdb_file_name, pdb_profile)
        # print(processed_atom_lines)

