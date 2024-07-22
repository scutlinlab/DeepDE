import os
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score, ndcg_score

import config 
from low_n_utils import modules
from low_n_utils import rep_manage
from low_n_utils import A003_common

choose_step = "step1"
step2_use_bright = 0
classfy_score = 1.0

def calc_ndcg(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / y_true.std() 
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def get_classfy(qfunc_list, classfy_lev):
    class_list = []
    for qfunc in qfunc_list:
        if qfunc >= classfy_lev:
            class_list.append(2)
        else:
            class_list.append(1)
    class_array = np.array(class_list)
    return class_array

def main(args):
    classfy_model = args.classfy_model
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    sele_low_n_df = modules.SelectLowNDataFram(config, args)
    # sub_train_df = sele_low_n_df.select_train_df()
    if choose_step == "step2":
        sub_train_df_all = pd.read_csv(f"/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_{args.seed}.csv")
        if step2_use_bright:
            print("step2_use_bright_train_set!")
            sub_train_df_all.drop(sub_train_df_all.columns[[0]], axis=1,inplace=True)
            sub_train_df = sub_train_df_all[sub_train_df_all["quantitative_function"] >= 0.6]
            print("bright_train_num: ", len(list(sub_train_df["quantitative_function"])))
        else:
            print("step2_use_normal_train_set!")
            sub_train_df = sub_train_df_all
            print("normal_train_num: ", len(list(sub_train_df["quantitative_function"])))
    elif choose_step == "step1":
        print("step1_use_normal_train_set!")
        sub_train_df = pd.read_csv(f"/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_{args.n_train_seqs}_{args.seed}.csv")
        sub_train_df.drop(sub_train_df.columns[[0]], axis=1,inplace=True)
        print("normal_train_num: ", len(list(sub_train_df["quantitative_function"])))
    # sub_train_df.to_csv(f"/share/jake/Low_N_data/train_csv/{args.training_objectives}_{args.model_name}_train_{args.sampling_method}_1000_{args.seed}.csv")
    train_qfunc = sele_low_n_df.get_qfunc(sub_train_df)
    train_classfy_qfunc = get_classfy(train_qfunc, 1.0)
    

    print(sub_train_df.head())
    # print(sub_train_df['distance'])
    rep_mana = rep_manage.RepManage(config, args)
    train_reps = rep_mana.get_train_reps(sub_train_df)
    print(f"{args.model_name} train ebd: ", train_reps[0])
    print(f"{args.model_name} train ebd shape: ", train_reps.shape)
    print("train nums: ", len(list(train_qfunc)))

    gener_top_model = modules.GenerateTopModel(config, args)
    # top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
    if choose_step == "step1":
        if "LDA" in classfy_model:
            lda = LDA()
            train_reps_transform = lda.fit_transform(train_reps, train_classfy_qfunc)
            if "LR" in classfy_model:
                print(classfy_model)
                top_model = A003_common.train_logistic_regression(train_reps_transform, train_classfy_qfunc)
            elif "SVM" in classfy_model:
                if "polynomial" in classfy_model:
                    print(classfy_model)
                    top_model = A003_common.trainn_svm_polynomial(train_reps_transform, train_classfy_qfunc)
                elif "rbf" in classfy_model:
                    print(classfy_model)
                    top_model = A003_common.trainn_svm_rbf(train_reps_transform, train_classfy_qfunc)
                elif "lin" in classfy_model:
                    print(classfy_model)
                    top_model = A003_common.train_svm_linear(train_reps_transform, train_classfy_qfunc)
        elif classfy_model == "LR":
            print(classfy_model)
            top_model = A003_common.train_logistic_regression(train_reps, train_classfy_qfunc)
        elif "SVM" in classfy_model:
            if "polynomial" in classfy_model:
                print(classfy_model)
                top_model = A003_common.trainn_svm_polynomial(train_reps, train_classfy_qfunc)
            elif "lin" in classfy_model:
                print(classfy_model)
                top_model = A003_common.train_svm_linear(train_reps, train_classfy_qfunc)
            elif "rbf" in classfy_model:
                print(classfy_model)
                top_model = A003_common.trainn_svm_rbf(train_reps, train_classfy_qfunc)
    elif choose_step == "step2":
        top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
    # print(DASFADFJKADDHFJKASDFHASD)
    print(args.do_test)
    if args.do_test == 'True':
        if choose_step == "step2":
            test_df = pd.read_csv(f"/share/jake/hot_spot/data/test/classfy_test_result/LDA/step1/{classfy_model}_classification{classfy_score}_{args.model_name}_random_1-2_gfp_SK_test_3_{args.seed}.csv")
            test_df.drop(test_df.columns[[0]], axis=1,inplace=True)
            test_qfunc = sele_low_n_df.get_qfunc(test_df)
        elif choose_step == "step1":
            test_df = sele_low_n_df.select_test_df()
            test_qfunc = np.array(test_df[f"classification {classfy_score}"])
        test_reps = rep_mana.get_test_reps(test_df)
        print(test_reps.shape)
        # yhat = rep_mana.predict_rep_fitness(top_model, test_reps)
        
        if choose_step == "step2":
            yhat = rep_mana.predict_rep_fitness(top_model, test_reps)
            test_df["predict_qfunc_step2"] = list(yhat)
            if args.save_test_result == 'True':
                if step2_use_bright:
                    test_df.to_csv(f"/share/jake/hot_spot/data/test/classfy_test_result/LDA/step2/step1_{classfy_model}_classification{classfy_score}_{args.model_name}_random_1-2_gfp_SK_test_3_step2_lin_bright_{args.seed}.csv")
                else:
                    test_df.to_csv(f"/share/jake/hot_spot/data/test/classfy_test_result/LDA/step2/step1_{classfy_model}_classification{classfy_score}_{args.model_name}_random_1-2_gfp_SK_test_3_step2_lin_{args.seed}.csv")
        elif choose_step == "step1":
            if "LDA" in classfy_model:
                test_reps_transform = lda.transform(test_reps)
                yhat = top_model.predict(test_reps_transform)
            else:
                yhat = top_model.predict(test_reps)
            print(yhat)
            # print(dafadfasdfasdfasdfgas)
            test_df[f"predict_classfy_{classfy_score}"] = list(yhat)
            if args.save_test_result == 'True':
                test_df.to_csv(f"/share/jake/hot_spot/data/test/classfy_test_result/LDA/step1/{classfy_model}_classification{classfy_score}_{args.model_name}_random_1-2_gfp_SK_test_3_{args.seed}.csv")
        print(len(test_qfunc), len(yhat))
        r = stats.pearsonr(test_qfunc, yhat)[0]
        rho = stats.spearmanr(test_qfunc, yhat).correlation
        r2 = r2_score(test_qfunc, yhat)
        ndcg = calc_ndcg(yhat, test_qfunc)
        pred_and_real = np.array(list(yhat) + list(test_qfunc))
        name_list = []
        for i in range(len(test_qfunc)):
            name_list.append(i)
        print("r : ", r, "rho : ", rho, "r2 : ", r2, "ndcg : ", ndcg)
        

                
    if args.do_design == 'True':
        assert args.use_bright
        assert choose_step == "step2"
        seed = 0
        design_site_model = "eUniRep"
        method = "method2"
        train_seq_num = "1000"
        objective = "top_1%"
        assert seed == args.seed
        if method == "method2":
            input_dair_path = f"/share/jake/hot_spot/data/result/method_2/design_seqs_result/{design_site_model}/all_3_mutation_1-2_{objective}_result_{train_seq_num}_{seed}_new"
            output_dair_path = f"/share/jake/hot_spot/data/result/method_3/design_seqs_result/{design_site_model}_design/{method}/{args.model_name}/all_3_mutation_1-2_{objective}_result_{train_seq_num}_{seed}"
        elif method == "method1":
            input_dair_path = f"/share/jake/hot_spot/data/result/method_1/design_seqs_result/{design_site_model}/all_3_mutation_{objective}_result_{train_seq_num}_{seed}_new"
            output_dair_path = f"/share/jake/hot_spot/data/result/method_3/design_seqs_result/{design_site_model}_design/{method}/{args.model_name}/all_3_mutation_{objective}_result_{train_seq_num}_{seed}"
        # input_dair_path = f"/share/jake/hot_spot/data/result/method_1/design_seqs/temporary"
        # output_dair_path = f"/share/jake/hot_spot/data/result/method_1/design_seqs_result/temporary"
        all_file_name = os.listdir(input_dair_path)
        for file_name in tqdm(all_file_name):
            input_path = f"{input_dair_path}/{file_name}"
            df_1 = pd.read_csv(input_path)
            df_1.drop(df_1.columns[[0]], axis=1,inplace=True)
            df = df_1[df_1["predict_qfunce"] > 0.6]
            df = df.copy()
            rep_array = rep_mana.generate_uni_reps(list(df["seq"]))
            yhat = rep_mana.predict_rep_fitness(top_model, rep_array)
            # yhat_df, yhat_std, yhat_mem = rep_mana.get_fitness(list(df["seqs"]), top_model)
            df[f"{args.model_name}_predict_qfunce_step2"] = list(yhat)
            df.sort_values(by = f"{args.model_name}_predict_qfunce_step2", ascending=False, inplace = True)
            df = df.reset_index(drop = True)
            df.to_csv(f"{output_dair_path}/{file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int,
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="""可以指定使用服务器上的特定GPU."""
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="""可以选择使用哪种表征方法进行辅助定向进化,现有的表征方法有 -UniRep -eUniRep -Random_UniRep."""
    )
    parser.add_argument(
        "--training_objectives", type=str, default=None,
        help="""可以选择特定的测试目标,现有的测试目标包括 -gfp_34536 -gfp_34536_split"""
    )
    parser.add_argument(
        "--do_design", type=str, default='False',
        help="""可以选择是否进行蛋白质的设计步骤""",
        required=True
    )
    parser.add_argument(
        "--do_test", type=str, default='True',
        help="""可以指定是否使用特定的测试集进行测试"""
    )
    parser.add_argument(
        "--save_test_result", type=str, default='False',
        help="""可以选择是否保存测试结果"""
    )
    parser.add_argument(
        "--n_train_seqs", type=int, default=None,
        help="""确定训练集数量"""
    )
    parser.add_argument(
        "--sampling_method", type=str, default=None,
        help="""确定如何生成训练集"""
    )
    parser.add_argument(
        "--top_model_name", type=str, default=None,
        help="""确定使用何种下游模型"""
    )
    parser.add_argument(
        "--use_bright", type=int, default=0,
        help="""是否只是使用亮的序列作为训练集"""
    )
    parser.add_argument(
        "--classfy_model", type=str, default=None,
        help="""确定step1使用哪种分类模型"""
    )

    args = parser.parse_args()

    if(args.seed is None):
        raise ValueError(
        "--seed must be specified"
                            )
    if(args.model_name is None):
        raise ValueError(
        "--model_name must be specified, you can choose -UniRep -eUniRep -Random_UniRep"
                            )
    if(args.training_objectives is None):
        raise ValueError(
        "--training_objectives must be specified, you can choose -gfp_34536 -gfp_34536_split"
                            )
    if(args.gpu is None):
        raise ValueError(
        "--gpu must be specified"
                            )
    if(args.do_design is None):
        raise ValueError(
        "--do_design must be specified"
                            )
    if(args.do_test is None):
        raise ValueError(
        "--do_test must be specified"
                            )
    if(args.save_test_result is None):
        raise ValueError(
        "--save_test_result must be specified"
                            )
    main(args)