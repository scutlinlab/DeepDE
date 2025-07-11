import itertools
import Levenshtein
import pandas as pd
from tqdm import tqdm

gfp_wt = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
AA_LIST = ['A', 'C', 'D', 'E', 'F', 
            'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 
            'S', 'T', 'V', 'W', 'Y']

def make_name(pos_list):
    name = ""
    for i, pos in enumerate(pos_list):
        name += str(pos)
        if i < len(pos_list) - 1:
            name += " "
    return name

def sum_list(list, size):
    if (size == 0):
        return 0
    else:
        return list[size - 1] + sum_list(list, size - 1)

def combination_2 (combination_list):
    list_1 = []
    for chr_1 in combination_list:
        for chr_2 in combination_list: 
            list_1.append((chr_1, chr_2))
    return list_1

def combination_3(combination_list):
        list_1 = []
        for chr_1 in combination_list:
            for chr_2 in combination_list:
                for chr_3 in combination_list:
                    list_1.append((chr_1, chr_2, chr_3))
        return list_1

def mutation_pos (target_seq, mutation_num):
    seq_pos = []
    mut_site = []
    distance = Levenshtein.distance(gfp_wt, target_seq)
    assert distance == mutation_num
    for i in range(len(target_seq)):
        if gfp_wt[i] != target_seq[i]:
            seq_pos.append(i)
            mut_site.append(target_seq[i])
    for i, pos in enumerate(seq_pos):
        assert target_seq[pos] == "_"
        assert mut_site[i] == "_"
    return seq_pos

def saturation_mutationv(mutation_pos, template_seq):
    mut_pos_list = []
    for pos in mutation_pos.split():
        mut_pos_list.append(int(pos))
    mutation_list = []
    for amino_list in combination_3(AA_LIST):
        seq = list(template_seq)
        for num in range(len(amino_list)):
            # print(seq[mut_pos_list[num]])
            assert seq[mut_pos_list[num]] == "_"
            seq[mut_pos_list[num]] = amino_list[num]
        if Levenshtein.distance(gfp_wt, ''.join(seq)) == 3:
            mutation_list.append(''.join(seq))
    return mutation_list

def generate_mut_seqs(template_seq):
    mutation_list = []
    mut_pos_list = mutation_pos(template_seq, 2)
    for amino_list in list(combination_2(AA_LIST)):
        seq = list(template_seq)
        for num in range(len(amino_list)):
            assert seq[mut_pos_list[num]] == "_"
            seq[mut_pos_list[num]] = amino_list[num]
        if Levenshtein.distance(gfp_wt, ''.join(seq)) == 2:
            mutation_list.append(''.join(seq))
    name_list = []
    for h in range(len(mutation_list)):
        name_list.append(h)
    print(len(name_list))
    assert len(list(dict.fromkeys(mutation_list))) == 361
    return mutation_list


def stati_triple_mut(all_2_mutation_df:pd.DataFrame, all_3_mutation_df:pd.DataFrame, stati_target:str):
    # all_3_mutation_df.drop(all_3_mutation_df.columns[[0]], axis=1,inplace=True)
    datas_name_3 = all_3_mutation_df.columns.tolist()
    datas_name_3.insert(len(datas_name_3), stati_target)

    print(datas_name_3)

    all_3_names = list(all_3_mutation_df["name"])
    target_pre_mut_list = []

    for name in tqdm(all_3_names):
        pos_list = name.split()
        top_list = []
        
        for double_pos in list(itertools.combinations(pos_list, 2)):
            double_name = make_name(double_pos)
            df_2 = all_2_mutation_df[(all_2_mutation_df["name"] == double_name)]
            top_list.append(float(df_2[stati_target]))

        top_predict = sum_list(top_list, 3) / 3
        target_pre_mut_list.append(top_predict)

    all_3_mutation_df[stati_target] = target_pre_mut_list
    return all_3_mutation_df

def stati_triple_mut_old(all_2_mutation_df:pd.DataFrame, all_3_mutation_df:pd.DataFrame, stati_target:str):
    all_3_mutation_df.drop(all_3_mutation_df.columns[[0]], axis=1,inplace=True)
    datas_name_3 = all_3_mutation_df.columns.tolist()
    datas_name_3.insert(len(datas_name_3), stati_target)

    print(datas_name_3)

    all_3_names = list(all_3_mutation_df["name"])
    for name in tqdm(all_3_names):
        index = all_3_mutation_df[(all_3_mutation_df["name"] == name)].index.tolist()[0]
        pos_list = name.split()
        top1_list = []
        for double_pos in list(itertools.combinations(pos_list, 2)):
            double_name = make_name(double_pos)
            df_2 = all_2_mutation_df[(all_2_mutation_df["name"] == double_name)]
            # print(df_2)
            top1_list.append(float(df_2[stati_target]))

        top1_predict = sum_list(top1_list, 3) / 3
        
        all_3_mutation_df.loc[index, stati_target] = top1_predict
    return all_3_mutation_df
# all_3_mutation_df.to_csv("/share/jake/hot_spot/data/result/method_2/gfp_3_mutation_all.csv")

def design_seq_with_hotspot(all_3_mutation_df:pd.DataFrame, stati_target:str, target_num, output_path:str):

    print(f"Use top {target_num} mutations!")
    all_3_mutation_df.sort_values(by = stati_target, ascending=False, inplace = True)
    all_3_mutation_df = all_3_mutation_df.reset_index(drop = True)
    top_hotspot_df = all_3_mutation_df[:target_num]
    print(top_hotspot_df.head())

    top_hotspot_name_list = list(top_hotspot_df["name"])
    for name in tqdm(top_hotspot_name_list[0:10]):
        df_1 = top_hotspot_df[(top_hotspot_df["name"] == name)]
        template_seq = list(df_1["seqs"])[0]
        # print(template_seq)
        mutation_pos = name
        mutation_list = saturation_mutationv(mutation_pos, template_seq)
        template_seq_list = []
        name_list = []
        for i in range(len(mutation_list)):
            name_list.append(str(i))
            template_seq_list.append(template_seq)
        df_result = pd.DataFrame({"name": name_list, "seq": mutation_list, "template_seq": template_seq_list})
        df_result.to_csv(f"{output_path}/{name}.csv")
    return top_hotspot_df

def generate_seq_from_mut_site(temp_seq, mut_name):
    seq_list = generate_mut_seqs(temp_seq)
    names = []
    for i in range(len(seq_list)):
        names.append(mut_name)
    df = pd.DataFrame({"name": names, "seq": seq_list})
    return df