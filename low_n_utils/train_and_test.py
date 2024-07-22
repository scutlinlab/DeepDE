import random
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score, ndcg_score

# fixed_train_csv_path = "/home/wangqihan/low-N-protein-engineering-master/data/s3/datasets/tts_splits/data_distributions/sarkisyan_split_1_distance.csv"
gfp_wt = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
def create_dir_not_exist(path):
        if not os.path.exists(path):
            os.mkdir(path)

def calc_ndcg(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / y_true.std() 
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:#np.var函数的作用为求方差！
        return 0.0
    return stats.spearmanr(y_pred, y_true).correlation

def pearson(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:#np.var函数的作用为求方差！
        return 0.0
    return stats.pearsonr(y_pred, y_true)[0]

def r2(y_pred, y_true):
    return r2_score(y_true, y_pred)

def sum_list(list, size):
    if (size == 0):
        return 0
    else:
        return list[size - 1] + sum_list(list, size - 1)

def compare_seq(wt_seq, target_seq):
    wt_seq = list(wt_seq)
    target_seq = list(target_seq)
    assert len(wt_seq) == len(target_seq)
    mutation_pos = []
    original_amino = []
    mutation_amino = []
    for i in range(len(wt_seq)):
        if wt_seq[i] != target_seq[i]:
            mutation_pos.append(i)
            original_amino.append(wt_seq[i])
            mutation_amino.append(target_seq[i])
    return mutation_pos, original_amino, mutation_amino


def recall_statistics(result_df, pred_val_name):
    y_pred = list(result_df[pred_val_name])
    y_true = list(result_df.quantitative_function.values)
    # name_list = list(result_df.name.values)
    name_list = []
    for i in range(len(y_true)):
        name_list.append(i)
    assert len(name_list) == len(y_pred) == len(y_true)
    pred_and_real = np.array(y_pred + y_true)
    output_data = np.array([pred_and_real, name_list], dtype=object)
    return output_data

def get_classfy_num(dataframe, index):
    sort_num = 100
    dataframe.sort_values(by = index, ascending=False, inplace = True)
    dataframe = dataframe.reset_index(drop = True)
    df_bright_sort = dataframe[:sort_num]
    # print(list(df_bright_sort["quantitative_function"]))
    num_1 = 0
    num_105 = 0
    num_11 = 0
    num_6 = 0
    for i in list(df_bright_sort["quantitative_function"]):
        if i > 1:
            num_1 += 1
        if i > 1.05:
            num_105 += 1
        if i > 1.1:
            num_11 += 1
        if i < 0.6:
            num_6 += 1
    # print("calssfy 1: " ,num_1, num_1 / sort_num)
    # print("calssfy 0.9: " ,num_9, num_9 / sort_num)
    # print("calssfy 1.1: " ,num_11, num_11 / sort_num)
    return num_1, num_105, num_11, num_6

def recall_save(args, result_df, output_path):
    if args.top_model_name == "inference":
        create_dir_not_exist(os.path.join(output_path, "recall", args.top_model_name, f"{args.training_objectives}"))
        recall_output_dir = os.path.join(output_path, "recall", args.top_model_name, f"{args.training_objectives}", f"{args.model_name}")
        recall_output_name = f"{args.model_name}_{args.training_objectives}_seed_{args.seed}.npy"
    else:
        create_dir_not_exist(os.path.join(output_path, "recall", args.top_model_name, f"{args.training_objectives}_{args.n_train_seqs}"))
        recall_output_dir = os.path.join(output_path, "recall", args.top_model_name, f"{args.training_objectives}_{args.n_train_seqs}", f"{args.model_name}_{args.sampling_method}")
        recall_output_name = f"{args.model_name}_{args.sampling_method}_train_num_{args.n_train_seqs}_{args.training_objectives}_seed_{args.seed}.npy"
    create_dir_not_exist(recall_output_dir)
    pred_val_name = f"{args.model_name}_{args.top_model_name}"
    output_data = recall_statistics(result_df, pred_val_name)
    np.save(os.path.join(recall_output_dir, recall_output_name), output_data)
    

def train_and_test_select_low_n(args, config):
    np.random.seed(args.seed)
    random.seed(args.seed)
    target_protein = args.training_objectives.split("_")[0]
    test_name = args.training_objectives.split("_")[1]
    train_set_path = os.path.join(config.DATA_SET_PATH, "train_csv", target_protein, "sk_train_set_distance.csv")
    test_set_path = os.path.join(config.DATA_SET_PATH, "test_csv", target_protein, f"{test_name.lower()}_test_set_distance.csv")
    train_df_all = pd.read_csv(train_set_path)
    
    if len(args.sampling_method.split("_")) == 1:
        assert args.sampling_method.split("_")[0] == "random"
        train_df_mut = train_df_all
    else:
        train_df_mut = pd.DataFrame()
        assert args.sampling_method.split("_")[0] == "random"
        train_mutation = args.sampling_method.split("_")[1]
        if len(train_mutation) > 1:
            train_mutation_list = list(range(int(train_mutation.split("-")[0]), int(train_mutation.split("-")[1]) + 1))
        else:
            train_mutation_list = [train_mutation]
        for num in train_mutation_list:
            mutation_num = int(num)
            df = train_df_all[train_df_all["distance"] == mutation_num]
            train_df_mut = pd.concat([train_df_mut, df])
    train_df = train_df_mut.sample(n=args.n_train_seqs)
    test_df_temp = pd.read_csv(test_set_path)
    
    if "test" in args.training_objectives:
        test_df = pd.DataFrame()
        test_mutation = args.training_objectives.split("_")[-1]
        if len(test_mutation) > 1:
            test_mutation_list = list(range(int(test_mutation.split("-")[0]), int(test_mutation.split("-")[1]) + 1))
        else:
            test_mutation_list = [test_mutation]
        for num in test_mutation_list:
            mutation_num = int(num)
            df = test_df_temp[test_df_temp["distance"] == mutation_num]
            test_df = pd.concat([test_df, df])
    else:
        test_df = test_df_temp
    if "split" in args.training_objectives:
        test_df_all = test_df.copy()
        np.random.seed(0)
        random.seed(0)
        if "SN" in args.training_objectives:
            df_high_function = test_df_all[test_df_all['quantitative_function'] > 1.5].sample(n=96)
            df_low_function = test_df_all[test_df_all['quantitative_function'] < 0.7]
        elif "SK" in args.training_objectives:
            df_high_function = test_df_all[test_df_all['quantitative_function'] > 1.0].sample(n=96)
            df_low_function = test_df_all[test_df_all['quantitative_function'] < 0.6]
        test_df = pd.concat([df_high_function, df_low_function])
    return train_df, test_df

def train_and_test_select(args, config):
    np.random.seed(args.seed)
    random.seed(args.seed)
    target_protein = args.training_objectives.split("_")[0]
    test_name = args.training_objectives.split("_")[1]
    input_csv_path = os.path.join(config.DATA_SET_PATH, "csv", target_protein, f"{test_name.lower()}_data_set_distance.csv")
    all_datas = pd.read_csv(input_csv_path)
    # print(len(list(all_datas[all_datas["distance"] <= 3]["distance"])))#dbug
    if len(args.sampling_method.split("_")) == 1:
        assert args.sampling_method.split("_")[0] == "random"
        train_df_all = all_datas
    else:
        train_df_all = pd.DataFrame()
        assert args.sampling_method.split("_")[0] == "random"
        train_mutation = args.sampling_method.split("_")[1]
        if len(train_mutation) > 1:
            train_mutation_list = list(range(int(train_mutation.split("-")[0]), int(train_mutation.split("-")[1]) + 1))
        else:
            train_mutation_list = [train_mutation]
        for num in train_mutation_list:
            mutation_num = int(num)
            df = all_datas[all_datas["distance"] == mutation_num]
            train_df_all = pd.concat([train_df_all, df])
    if args.use_bright:
        train_df_all = train_df_all[train_df_all["quantitative_function"] >= 0.705]
        print("---------------use_bright !--------------")

    train_df = train_df_all.sample(n=args.n_train_seqs)
    rest_df = all_datas.drop(train_df.index)
   
    if "test" in args.training_objectives:
        test_df_all = pd.DataFrame()
        test_mutation = args.training_objectives.split("_")[-1]
        if len(test_mutation) > 1:
            test_mutation_list = list(range(int(test_mutation.split("-")[0]), int(test_mutation.split("-")[1]) + 1))
        else:
            test_mutation_list = [test_mutation]
        for num in test_mutation_list:
            mutation_num = int(num)
            df = rest_df[rest_df["distance"] == mutation_num]
            test_df_all = pd.concat([test_df_all, df])
    else:
        test_df_all = rest_df
    if "split" in args.training_objectives:
        np.random.seed(0)
        random.seed(0)
        if "SN" in args.training_objectives:
            df_high_function = test_df_all[test_df_all['quantitative_function'] > 1.5].sample(n=96)
            df_low_function = test_df_all[test_df_all['quantitative_function'] < 0.7]
        elif "SK" in args.training_objectives:
            df_high_function = test_df_all[test_df_all['quantitative_function'] > 1.0].sample(n=96)
            df_low_function = test_df_all[test_df_all['quantitative_function'] < 0.6]
        test_df = pd.concat([df_high_function, df_low_function])
    else:
        test_df = test_df_all.copy()
    return  train_df, test_df

def result_statistics(args, config, result_df):
    output_path = config.OUTPUT_PATH
    if args.save_test_result == "True":
        recall_save(args, result_df, output_path)
        predict_qfunc_save_path = os.path.join(output_path, "predict_qfunc", args.model_name)
        create_dir_not_exist(predict_qfunc_save_path)
        if args.top_model_name == "inference":
            result_df.to_csv(f"{predict_qfunc_save_path}/{args.model_name}_{args.top_model_name}_{args.training_objectives}_seed_{args.seed}.csv")
        else:
            result_df.to_csv(f"{predict_qfunc_save_path}/{args.model_name}_{args.top_model_name}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_seed_{args.seed}.csv")
    metric_fns = {
        'pearsonr': pearson,
        'spearman': spearman,
        'ndcg': calc_ndcg,
        'r2': r2
    }
    results_dict = {k: mf(np.array(result_df[f"{args.model_name}_{args.top_model_name}"]), np.array(result_df["quantitative_function"]))
            for k, mf in metric_fns.items()}
    print("results: ", results_dict)
    if "split" not in args.training_objectives:
        
        if "test" in args.training_objectives:
            test_mutation = args.training_objectives.split("_")[-1]
            if len(test_mutation) > 1:
                start = int(test_mutation.split("-")[0])
                end = int(test_mutation.split("-")[1]) + 1
            else:
                start = end = 1
        else:
            start = 1
            end = 5

        for j in range(start, end):
            y_pred = np.array(result_df[result_df.distance == j][f"{args.model_name}_{args.top_model_name}"])
            y_true = np.array(result_df[result_df.distance == j].quantitative_function.values)
            results_dict.update({
                    f'{k}_{j}mut': mf(y_pred, y_true)
                    for k, mf in metric_fns.items()})
    if args.top_model_name == "inference":
        results_dict.update({
            'model_name': args.model_name,
            "predict_type": args.top_model_name,
            'seed': args.seed
        })
        output_name = f"{args.model_name}_{args.top_model_name}_{args.training_objectives}_result_statistics.csv"
    else:
        results_dict.update({
            'model_name': args.model_name,
            "predict_type": args.top_model_name,
            'sampling_method': args.sampling_method,
            'n_train_seqs': args.n_train_seqs,
            'seed': args.seed
        })
        output_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_result_statistics.csv"
    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)
    if args.save_test_result == "True":
        create_dir_not_exist(os.path.join(output_path, "p_or_s", args.top_model_name, args.model_name))
        outpath = os.path.join(output_path, "p_or_s", args.top_model_name, args.model_name, output_name)
        if os.path.exists(outpath):
            results.to_csv(outpath, mode='a', header=False, index=False,
                    columns=sorted(results.columns.values))
        else:
            results.to_csv(outpath, mode='w', index=False,
                    columns=sorted(results.columns.values))
    return results

# def data_processing_method3_step1(args, test_df):
#     choose_step2_list = []
#     for predict_qfunc in list(test_df[f"{args.model_name}_{args.top_model_name}_step1"]):
#         if predict_qfunc > 0.6:
#             choose_step2_list.append(1)
#         else:
#             choose_step2_list.append(0)
#     test_df["step2_test_seqs"] = choose_step2_list
#     return test_df

def method3_result_statistics(args, config, test_df:pd.DataFrame, step1_bright_degree = 0.6, step2_bright_degree = 1.0):
    output_path = config.OUTPUT_PATH

    step1_recall_1, step1_recall_105, step1_recall_11, step1_predict_less_06 = get_classfy_num(test_df, f"{args.model_name}_{args.top_model_name}")
    results_dict = {
        "step 1 recall 1.0": step1_recall_1,
        "step 1 recall 1.05": step1_recall_105,
        "step 1 recall 1.1": step1_recall_11,
        "step 1 recall less 0.6": step1_predict_less_06,
    }
    if args.top_model_name != "inference":
        setp2_test_df = test_df[test_df[f"{args.model_name}_{args.top_model_name}"] > step1_bright_degree]
        print("Quantity of remaining parts fo step1(Quantity of step2 test df): ", setp2_test_df.shape[0])
        step1_classfy_06 = setp2_test_df[setp2_test_df["quantitative_function"] > step1_bright_degree].shape[0] / setp2_test_df.shape[0]
        step1_predict_hight = test_df[test_df[f"{args.model_name}_{args.top_model_name}"] > step2_bright_degree]
        step1_pre_postive = step1_predict_hight.shape[0]
        print("step1_pre_postive: ", step1_pre_postive, len(list(step1_predict_hight["quantitative_function"])))
        step1_true_postive_num = step1_predict_hight[step1_predict_hight["quantitative_function"] > step2_bright_degree].shape[0]
        step1_true_postive_11_num = step1_predict_hight[step1_predict_hight["quantitative_function"] > 1.1].shape[0]
        step1_false_postive_06_num = step1_predict_hight[step1_predict_hight["quantitative_function"] < 0.6].shape[0]
        if step1_pre_postive == 0:
            step1_true_postive = step1_true_postive_11 = step1_false_postive_06 = 0
        else:
            step1_true_postive = step1_true_postive_num / step1_pre_postive
            step1_true_postive_11 = step1_true_postive_11_num / step1_pre_postive
            step1_false_postive_06 = step1_false_postive_06_num / step1_pre_postive
        
        results_dict.update({
            "step 1 classfy 0.6": step1_classfy_06,
            "step 1 predict postive": step1_pre_postive,
            "step 1 true postive num": step1_true_postive_num,
            "step 1 true postive": step1_true_postive,
            "step 1 true postive num(>1.1)": step1_true_postive_11_num,
            "step 1 true postive(>1.1)": step1_true_postive_11,
            "step 1 true postive num(>1.1)": step1_true_postive_11_num,
            "step 1 true postive(>1.1)": step1_true_postive_11,
            "step 1 false postive num(<0.6)": step1_false_postive_06_num,
            "step 1 false postive(<0.6)": step1_false_postive_06,
        })

        step2_predict_hight = setp2_test_df[setp2_test_df[f"{args.model_name}_{args.top_model_name}_step2"] > step2_bright_degree]
        step2_pre_postive = step2_predict_hight.shape[0]
        step2_true_postive_num = step2_predict_hight[step2_predict_hight["quantitative_function"] > step2_bright_degree].shape[0]
        step2_true_postive_11_num = step2_predict_hight[step2_predict_hight["quantitative_function"] > 1.1].shape[0]
        step2_false_postive_06_num = step2_predict_hight[step2_predict_hight["quantitative_function"] < 0.6].shape[0]
        if step2_pre_postive == 0:
            step2_true_postive = step2_true_postive_11 = step2_false_postive_06 = 0
        else:
            step2_true_postive = step2_true_postive_num / step2_pre_postive
            step2_true_postive_11 = step2_true_postive_11_num / step2_pre_postive
            step2_false_postive_06 = step2_false_postive_06_num / step2_pre_postive


        
        step2_recall_1, step2_recall_105, step2_recall_11, step2_predict_less_06 = get_classfy_num(setp2_test_df, f"{args.model_name}_{args.top_model_name}_step2")

    
        results_dict.update({
            "step 2 predict postive": step2_pre_postive,
            "step 2 true postive num": step2_true_postive_num,
            "step 2 true postive": step2_true_postive,
            "step 2 true postive num(>1.1)": step2_true_postive_11_num,
            "step 2 true postive(>1.1)": step2_true_postive_11,
            "step 2 true postive num(>1.1)": step2_true_postive_11_num,
            "step 2 true postive(>1.1)": step2_true_postive_11,
            "step 2 false postive num(<0.6)": step2_false_postive_06_num,
            "step 2 false postive(<0.6)": step2_false_postive_06,
            "step 2 recall 1.0": step2_recall_1,
            "step 2 recall 1.05": step2_recall_105,
            "step 2 recall 1.1": step2_recall_11,
            "step 2 recall less 0.6": step2_predict_less_06
        })
    if args.save_test_result == "True":
        print("Save method 3 test qfunc and statistics result !")
        results = pd.DataFrame(columns=sorted(results_dict.keys()))
        results = results.append(results_dict, ignore_index=True)
        if args.top_model_name == "inference":
            test_df_output_name = f"{args.model_name}_{args.top_model_name}_{args.training_objectives}_seed_{args.seed}_predict_qfunc_method3.csv"
            output_name = f"{args.model_name}_{args.top_model_name}_{args.training_objectives}_result_method3.csv"
        else:
            test_df_output_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_seed_{args.seed}_predict_qfunc_method3.csv"
            output_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_result_method3.csv"
        predict_qfunc_save_path = os.path.join(output_path, "method_3", "predict_qfunc", args.model_name)
        create_dir_not_exist(predict_qfunc_save_path)
        test_df.to_csv(f"{predict_qfunc_save_path}/{test_df_output_name}")
    
        outpath = os.path.join(output_path, "method_3", "result", f"{args.top_model_name}", output_name)
        if os.path.exists(outpath):
            results.to_csv(outpath, mode='a', header=False, index=False,
                    columns=sorted(results.columns.values))
        else:
            results.to_csv(outpath, mode='w', index=False,
                    columns=sorted(results.columns.values))
    return results_dict

def method2_satistic(args, mutation_2_df:pd.DataFrame, all_2_mutation_df:pd.DataFrame):
    satistic_list = [100, 20, 10, 5]
    mut_2_df_num = mutation_2_df.shape[0]
    print("mut_2_df_num", mut_2_df_num)
    name = list(mutation_2_df["name"])[0]
    index = all_2_mutation_df[(all_2_mutation_df["name"] == name)].index.tolist()[0]
    if args.top_model_name == "inference":
        sort_name = f"{args.model_name}_inference"
    else:
        sort_name = f"{args.model_name}_predict"
    print(mutation_2_df.head())
    mutation_2_df.sort_values(by = sort_name, ascending=False, inplace = True)
    mutation_2_df = mutation_2_df.reset_index(drop = True)
    max_func = list(mutation_2_df[sort_name])[0]
    all_2_mutation_df.loc[index, "max_func"] = max_func
    for num in satistic_list:
        sort_num = mut_2_df_num // num
        top_pre_func = sum_list(list(mutation_2_df[sort_name])[0:sort_num], sort_num) / sort_num
        all_2_mutation_df.loc[index, f"top_{100 // num}%_func"] = top_pre_func
    return all_2_mutation_df

def method2_satistic_step2(args, mutation_2_df:pd.DataFrame, all_2_mutation_df:pd.DataFrame):
    satistic_list = [100, 20, 10, 5]
    mut_2_df_num = mutation_2_df.shape[0]
    print("mut_2_df_num", mut_2_df_num)
    name = list(mutation_2_df["name"])[0]
    index = all_2_mutation_df[(all_2_mutation_df["name"] == name)].index.tolist()[0]
    step2_mutation_2_df = mutation_2_df[mutation_2_df[f"{args.model_name}_predict"] > 0.6]
    df_step2_num = len(list(step2_mutation_2_df["name"]))
    sort_name = f"{args.model_name}_predict_step2"
    print(step2_mutation_2_df.head())
    step2_mutation_2_df.sort_values(by = sort_name, ascending=False, inplace = True)
    step2_mutation_2_df = step2_mutation_2_df.reset_index(drop = True)
    if df_step2_num == 0:
        max_func = 0
    else:
        max_func = list(step2_mutation_2_df[sort_name])[0]
    all_2_mutation_df.loc[index, "max_func"] = max_func
    for num in satistic_list:
        sort_num = mut_2_df_num // num
        if df_step2_num < sort_num:
            top_pre_func = 0
        else:
            top_pre_func = sum_list(list(step2_mutation_2_df[sort_name])[0:sort_num], sort_num) / sort_num
        all_2_mutation_df.loc[index, f"top_{100 // num}%_func"] = top_pre_func
    return all_2_mutation_df

def triple_mut_concat(doub_mut1:list, doub_mut2:list):
    triple_mut_list = list(set(doub_mut1 + doub_mut2))
    triple_mut_list.sort()
    all_doub_list = list(itertools.combinations(triple_mut_list, 2))
    all_doub_list.remove(tuple(doub_mut1))
    all_doub_list.remove(tuple(doub_mut2))
    doub_mut3 = all_doub_list[0]
    name_third = str(doub_mut3[0]) + " " + str(doub_mut3[1])
    return triple_mut_list, name_third

def triple_mut_design(all_2_mut_csv:pd.DataFrame, sort_target:str):
    doub_mut_name_list = list(all_2_mut_csv["name"])
    all_target_list = []
    all_fitness_list = []
    all_mut_seq_list = []
    # has_used_doub_muts = []
    while len(all_target_list) < 100 and len(doub_mut_name_list) > 0:
        name_first = doub_mut_name_list[0]
        # has_used_doub_muts.append(name_first)
        doub_mut_name_list.remove(name_first)
        name_first_list = [int(i) for i in name_first.split(" ")]
        fitness_first = list(all_2_mut_csv[all_2_mut_csv["name"] == name_first][sort_target])[0]
        for name_second in doub_mut_name_list:
            name_second_list = [int(i) for i in name_second.split(" ")]
            if name_second_list[0] in name_first_list or name_second_list[1] in name_first_list:
                # has_used_doub_muts.append(name_second)
                doub_mut_name_list.remove(name_second)
                fitness_second = list(all_2_mut_csv[all_2_mut_csv["name"] == name_second][sort_target])[0]
                triple_mut_list, name_third = triple_mut_concat(name_first_list, name_second_list)
                if name_third not in doub_mut_name_list:
                    break
                # print(name_first, " " ,name_second, " ", name_third)
                doub_mut_name_list.remove(name_third)
                fitness_third = list(all_2_mut_csv[all_2_mut_csv["name"] == name_third][sort_target])[0]
                
                if triple_mut_list[1] - triple_mut_list[0] >= 17 and triple_mut_list[2] - triple_mut_list[1] >= 17:
                    all_target_list.append(f"{triple_mut_list[0]} {triple_mut_list[1]} {triple_mut_list[2]}")
                    mut_seq = list(gfp_wt)
                    for pos in triple_mut_list:
                        mut_seq[pos] = "_"
                    all_mut_seq_list.append(''.join(mut_seq))
                    fitness = (fitness_first + fitness_second + fitness_third) / 3
                    all_fitness_list.append(fitness)
                break
    # print("The designed combination of 3 mutation sites is less than 100, the num of mut is: ", len(all_target_list))
    print("100 combinations of triple mutation sites were successfully designed")
    all_target_df = pd.DataFrame({"name": all_target_list, "seqs": all_mut_seq_list, sort_target: all_fitness_list})
    return all_target_df

def compare_des_seqs(args, df:pd.DataFrame):
    
    df.sort_values(by = f"{args.model_name}_{args.top_model_name}_predict", ascending=False, inplace = True)
    df = df.reset_index(drop = True)
    name_list = list(df["name"])
    seq_list = list(df["seq"])
    qfunc_list = list(df[f"{args.model_name}_{args.top_model_name}_predict"])
    # target_seq = list(df["seq"])[0]
    mutation_pos_list = []
    original_amino_list = []
    mutation_amino_list = []
    for target_seq in seq_list:
        mutation_pos, original_amino, mutation_amino = compare_seq(gfp_wt, target_seq)
        # print(mutation_pos, original_amino, mutation_amino)
        mutation_pos_str = ""
        original_amino_str = ""
        mutation_amino_str = ""
        for i in range(len(mutation_pos)):
            mutation_pos_str += str(mutation_pos[i])
            original_amino_str += str(original_amino[i])
            mutation_amino_str += str(mutation_amino[i])
            if i < len(mutation_pos) - 1:
                    mutation_pos_str += " "
                    original_amino_str += " "
                    mutation_amino_str += " "
        mutation_pos_list.append(mutation_pos_str)
        original_amino_list.append(original_amino_str)
        mutation_amino_list.append(mutation_amino_str)
    df_output = pd.DataFrame({"name": name_list, "seq": seq_list, f"{args.model_name}_{args.top_model_name}_predict": qfunc_list, "mutation_pos": mutation_pos_list, "original_amino": original_amino_list, "mutation_amino": mutation_amino_list})
    return df_output
