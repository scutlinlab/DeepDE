import pandas as pd
import os
import itertools

gfp_wt = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
model_name = "eUniRep-Augmenting_concate"#eUniRep-Augmenting_concate
top_model_name = "lin"
train_num = 1000
seed = 0
do_method3 = 0
all_2_mut_csv = pd.read_csv(f"/share/jake/github/low_n_output/method_2/all_2_mutation/{top_model_name}/{model_name}_lin_random_1-2_train_seqs_num_{train_num}_do_method3_{do_method3}_seed_{seed}_all_2_mut.csv")
# sort_target = "top_1%_func"#max_func top_1%_func
# all_2_mut_csv.sort_values(by = sort_target, ascending=False, inplace = True)
# all_2_mut_csv = all_2_mut_csv.reset_index(drop = True)
# print(all_2_mut_csv.head())
for sort_target in ["max_func", "top_1%_func", "top_5%_func", "top_10%_func", "top_20%_func"]:
    all_2_mut_csv.sort_values(by = sort_target, ascending=False, inplace = True)
    all_2_mut_csv = all_2_mut_csv.reset_index(drop = True)
    # print(all_2_mut_csv.head())
    # all_name_list = list(all_2_mut_csv["name"])[:2000]
    def triple_mut_concat(doub_mut1:list, doub_mut2:list):
        triple_mut_list = list(set(doub_mut1 + doub_mut2))
        triple_mut_list.sort()
        all_doub_list = list(itertools.combinations(triple_mut_list, 2))
        all_doub_list.remove(tuple(doub_mut1))
        all_doub_list.remove(tuple(doub_mut2))
        doub_mut3 = all_doub_list[0]
        name_third = str(doub_mut3[0]) + " " + str(doub_mut3[1])
        return triple_mut_list, name_third

    def triple_mut_design(all_2_mut_csv:pd.DataFrame):
        doub_mut_name_list = list(all_2_mut_csv["name"])
        all_target_list = []
        all_fitness_list = []
        all_mut_seq_list = []
        # has_used_doub_muts = []
        while len(all_target_list) < 100 and len(doub_mut_name_list) > 0:
            i = 0
            name_first = doub_mut_name_list[0]
            # has_used_doub_muts.append(name_first)
            doub_mut_name_list.remove(name_first)
            name_first_list = [int(i) for i in name_first.split(" ")]
            if name_first_list[1] - name_first_list[0] < 17:
                continue
            fitness_first = list(all_2_mut_csv[all_2_mut_csv["name"] == name_first][sort_target])[0]
            while i <= len(doub_mut_name_list) - 1:
                name_second = doub_mut_name_list[i]
                name_second_list = [int(i) for i in name_second.split(" ")]
                if name_second_list[1] - name_second_list[0] < 17:
                    doub_mut_name_list.remove(name_second)
                    continue
                if name_second_list[0] in name_first_list or name_second_list[1] in name_first_list:
                    print(name_first, " ", name_second)
                    # has_used_doub_muts.append(name_second)
                    # doub_mut_name_list.remove(name_second)
                    fitness_second = list(all_2_mut_csv[all_2_mut_csv["name"] == name_second][sort_target])[0]
                    triple_mut_list, name_third = triple_mut_concat(name_first_list, name_second_list)
                    if name_third not in doub_mut_name_list:
                        i += 1
                        continue
                    # print(name_first, " " ,name_second, " ", name_third)
                    # doub_mut_name_list.remove(name_third)
                    fitness_third = list(all_2_mut_csv[all_2_mut_csv["name"] == name_third][sort_target])[0]
                    if triple_mut_list[1] - triple_mut_list[0] >= 17 and triple_mut_list[2] - triple_mut_list[1] >= 17:
                        doub_mut_name_list.remove(name_second)
                        doub_mut_name_list.remove(name_third)
                        all_target_list.append(f"{triple_mut_list[0]} {triple_mut_list[1]} {triple_mut_list[2]}")
                        mut_seq = list(gfp_wt)
                        for pos in triple_mut_list:
                            mut_seq[pos] = "_"
                        all_mut_seq_list.append(''.join(mut_seq))
                        fitness = (fitness_first + fitness_second + fitness_third) / 3
                        all_fitness_list.append(fitness)
                        break
                    else:
                        i += 1
                else:
                    i += 1
        # print("The designed combination of 3 mutation sites is less than 100, the num of mut is: ", len(all_target_list))
        print("100 combinations of triple mutation sites were successfully designed")
        all_target_df = pd.DataFrame({"name": all_target_list, "seqs": all_mut_seq_list, sort_target: all_fitness_list})
        return all_target_df

    all_target_df = triple_mut_design(all_2_mut_csv)
    print(all_target_df.head())
    all_target_df.to_csv(f"/share/jake/github/low_n_output/method_2/new_method_top_100_3_mutation/{top_model_name}/{model_name}/new_method_top_100_{model_name}_{top_model_name}_random_1-2_train_num_{train_num}_do_method3_{do_method3}_seed_{seed}_{sort_target}_result.csv")