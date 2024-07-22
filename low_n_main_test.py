import os
import numpy as np
import pandas as pd
import torch
import argparse
from scipy import stats
from tqdm import tqdm
# from sklearn.metrics import r2_score

import config 
from low_n_utils import models
from low_n_utils import modules
from low_n_utils import rep_manage
from low_n_utils import statistics_triple_mutation as stat_tri_mut
from low_n_utils import train_and_test

from sklearn.metrics import r2_score, roc_auc_score, ndcg_score
import joblib

low_n_test_method = False
design_method = "method_2"
save_top_model = False
# use_saved_model = True

def generate_reps(args, df:pd.DataFrame):
    rep_inf = rep_manage.RepInference(config, args)
    if "Augmenting" in args.model_name:
        if "concate" in args.model_name:
            reps = rep_inf.concate(df)
        else:
            reps = rep_inf.Augmenting(df)
    else:
        reps = rep_inf.generate_uni_reps(df)
    return reps


def main(args):
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    rep_inf = rep_manage.RepInference(config, args)
    
    if low_n_test_method:
        sub_train_df, test_df = train_and_test.train_and_test_select_low_n(args, config)
    else:
        sub_train_df, test_df = train_and_test.train_and_test_select(args, config)
    # sub_train_df.to_csv("/share/jake/github/low-n_data/exp_data/train_set_1000.csv")
    
    all_exp_data = pd.read_csv("/share/jake/github/low-n_data/exp_data/all_exp_data_multiple_change_norm_distance.csv")
    sub_train_df = pd.concat([sub_train_df, all_exp_data])
    # sub_train_df.to_csv("/share/jake/github/low-n_data/exp_data/train_set_1438.csv")
    
    # sub_train_df = pd.read_csv("/share/jake/github/low-n_data/temp/SK_train_set_1000_exclude_seed_0.csv")
    # test_df = pd.read_csv("/share/jake/github/low-n_data/temp/SK_test_set_1000_exclude_seed_0.csv")
    # sub_train_df = pd.read_csv("/share/jake/github/low-n_data/temp/SK_train_set_1000_seed_0_new_mutation_pos.csv")
    # test_df = pd.read_csv("/share/jake/github/low-n_data/test_csv/gfp/experimental_data_new.csv")
    # test_df = pd.read_csv("/share/jake/github/low_n_output/method_2/design_seqs/lin_old/eUniRep-Augmenting_train_num_1000_seed_0_top_1%_do_method3_0_new_des_method_0/4 72 162.csv")
    # sub_train_df = pd.read_csv("/share/jake/github/low-n_data/temp/SK_train_set_1000_new_split_seed_0.csv")
    # sub_train_df = pd.read_csv("/home/wangqihan/github/nn4dms-master/low-n_train_set/Low-N_train_set_1000_seed_0_ratio.csv")
    # sub_train_df = pd.read_csv(f"/share/jake/github/low-n_data/train_csv/gfp/Low-N_train_set_1000_seed_{args.seed}_concat_54_exp_datas_ratio.csv")
    # test_df = pd.read_csv("/share/jake/github/low-n_data/temp/SK_test_set_all_3_mutations.csv")
    if args.top_model_name != "inference":
        train_qfunc = np.array(sub_train_df["quantitative_function"])
        print(sub_train_df.head())
        print("train seq num: ", train_qfunc.shape)
        train_reps = generate_reps(args, sub_train_df)
        print('train reps:', train_reps[0])
        print('shape of train reps:', train_reps.shape)
        gener_top_model = modules.GenerateTopModel(config, args)
        top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
        if args.do_method3:
            print("---Do method3 ! Use the bright part of the training set(qfunc > 0.6) !---")
            step2_train_df = sub_train_df[sub_train_df["quantitative_function"] >= 0.6]
            step2_train_qfunc = np.array(step2_train_df["quantitative_function"])
            step2_train_reps = generate_reps(args, step2_train_df)
            step2_top_model = gener_top_model.sele_top_model(step2_train_reps, step2_train_qfunc)
    else:
        rep_inf = rep_manage.RepInference(config, args)


    print("do_test: ", args.do_test)
    if args.do_test == 'True':
        print(test_df.head())
        # print("test rep num: ", len(list(test_df["quantitative_function"])))
        if args.top_model_name == "inference":
            test_reps_inference = rep_inf.generate_inference(test_df["seq"])
            yhat = list(np.squeeze(test_reps_inference))
        else:
            test_reps = generate_reps(args, test_df)
            print("shape of test rep: ", test_reps.shape)
            yhat, yhat_std, yhat_mem  = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
            print(yhat[0:10])
        # assert len(yhat) == len(list(test_df["quantitative_function"]))
        test_df[f"{args.model_name}_{args.top_model_name}"] = yhat
        # test_df.to_csv(f"/share/jake/github/low-n_data/temp/experimental_177_datas_eUniRep-Augmenting_train_set_1000_concat_54_exp_datas_result_seed_{args.seed}.csv.csv")
        # test_df.to_csv(f"/share/jake/github/low_n_output/predict_qfunc/eUniRep-Augmenting/eUniRep-Augmenting_lin_random_1-2_ratio_{args.training_objectives}_n_train_seqs_1000_seed_{args.seed}.csv")
        results = train_and_test.result_statistics(args, config, test_df)
        if args.do_method3:
            if args.top_model_name != "inference":
                step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(test_reps, return_std=True, return_member_predictions=True)
                test_df[f"{args.model_name}_{args.top_model_name}_step2"] = step2_yhat
            result_dict = train_and_test.method3_result_statistics(args, config, test_df)
            print(result_dict)

    print("do_design: ", args.do_design)
    if args.do_design == "True":
        print("Do design method is :", design_method)
        input_dair_path = "/share/jake/hot_spot/data/all_2_mutation"
        ori_output_paht = os.path.join(config.OUTPUT_PATH, "method_2", "original_result", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_do_method3_{args.do_method3}_seed_{args.seed}")
        all_2_mut_output_path = os.path.join(config.OUTPUT_PATH, "method_2", "all_2_mutation", args.top_model_name)
        top_1000_3_mut_output_path = os.path.join(config.OUTPUT_PATH, "method_2", "all_3_mutation", args.top_model_name)
        if not os.path.exists(ori_output_paht):
            os.mkdir(ori_output_paht)
        all_file_name = os.listdir(input_dair_path)#debug
        all_2_mutation_df = pd.read_csv("/share/jake/hot_spot/data/gfp_2_mutation_all.csv")
        for i, file_name in enumerate(all_file_name):
            print(f"Running inference on site combination {file_name}, num {i}/{len(all_file_name)}!")
            df = pd.read_csv(f"{input_dair_path}/{file_name}")
            if args.top_model_name == "inference":
                all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_do_method3_{args.do_method3}_seed_{args.seed}_all_2_mut.csv"
                design_reps_inference = rep_inf.generate_inference(df["seq"])
                design_yhat = list(np.squeeze(design_reps_inference))
                df[f"{args.model_name}_inference"] = design_yhat
            else:
                all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_seqs_num_{args.n_train_seqs}_do_method3_{args.do_method3}_seed_{args.seed}_all_2_mut.csv"
                design_reps = generate_reps(args, df)
                
                design_yhat, yhat_std, yhat_mem  = top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                df[f"{args.model_name}_predict"] = design_yhat

                if args.do_method3:
                    design_step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                    df[f"{args.model_name}_predict_step2"] = design_step2_yhat
            df.to_csv(f"/share/jake/github/low_n_output/method_2/original_result/lin/eUniRep-Augmenting_train_num_1000_concat_exp_data_multiple_change_norm_do_method3_0_seed_0/{file_name}")
            # df.to_csv(f"/share/jake/github/low_n_output/method_2/original_result/lin/eUniRep-Augmenting_concate_train_num_1000_do_method3_0_new_split/{file_name}")
            assert df.shape[0] == 361
            if args.do_method3:
                all_2_mutation_df = train_and_test.method2_satistic_step2(args, df, all_2_mutation_df)
            else:
                all_2_mutation_df = train_and_test.method2_satistic(args, df, all_2_mutation_df)
        all_2_mutation_df.to_csv("/share/jake/github/low-n_data/temp/SK_train_num_1000_concat_all_exp_data_multiple_change_norm_all_2_mutation_ratio.csv")
        # all_2_mutation_df.to_csv("/share/jake/github/low-n_data/temp/SK_train_num_1000_new_split_all_2_mutation.csv")
    
    if args.predict_design_seqs:

        des_method = "train_num_1000_concat_exp_data_multiple_change_norm_fix_4N 33V 72H 89K 116G 141Q 162G 170V 205V_to_12_mut_Top23_all_comb"#new_method_top_100 lim_pos_top_10

        if design_method == "method_2":
            # classify_method_list = ["top_20%", "top_10%", "top_5%", "top_1%", "max"]
            classify_method_list = ["top_1%"]
            for cls_method in tqdm(classify_method_list):
                input_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}_do_method3_{args.do_method3}_des_method_{des_method}")
                output_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs_result", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}_do_method3_{args.do_method3}_des_method_{des_method}")
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                all_design_names = os.listdir(input_path)
                for design_name in all_design_names:
                    pos_name = design_name.split(".")[0]
                    print(pos_name)
                    des_df = pd.read_csv(f"{input_path}/{design_name}")
                    if args.top_model_name == "inference":
                        des_reps_inference = rep_inf.generate_inference(des_df["seq"])
                        yhat = list(np.squeeze(des_reps_inference))
                    else:
                        des_reps = generate_reps(args, des_df)
            
                        print("shape of test rep: ", des_reps.shape)
                        yhat, yhat_std, yhat_mem  = top_model.predict(des_reps, return_std=True, return_member_predictions=True)
                        print(yhat)
                    des_df[f"{args.model_name}_{args.top_model_name}_predict"] = yhat
                    des_df = train_and_test.compare_des_seqs(args, des_df)
                    if args.do_method3:
                        step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(des_reps, return_std=True, return_member_predictions=True)
                        des_df[f"{args.model_name}_{args.top_model_name}_predict_step2"] = step2_yhat
                        des_df = des_df[des_df[f"{args.model_name}_{args.top_model_name}_predict"] > 0.6]
                        des_df.sort_values(by = f"{args.model_name}_{args.top_model_name}_predict_step2", ascending=False, inplace = True)
                        des_df = des_df.reset_index(drop = True)
                    # des_df = des_df[(des_df["mutation_pos"] == pos_name)]
                    des_df.to_csv(f"{output_path}/{design_name}")
                    

                
            







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
        "--do_method3", type=int, default=0,
        help="""是否执行方法3"""
    )
    parser.add_argument(
        "--predict_design_seqs", type=int, default=0,
        help="""是否对设计出的序列进行预测"""
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