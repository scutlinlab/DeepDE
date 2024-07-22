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
use_saved_model = False

def generate_reps(args, df:pd.DataFrame): #根据参数 args.model_name 选择的不同表征模型，对输入的蛋白质序列进行表征。
    rep_inf = rep_manage.RepInference(config, args)
    if "Augmenting" in args.model_name:
        if "concate" in args.model_name:
            reps = rep_inf.concate(df)
        else:
            reps = rep_inf.Augmenting(df)
    else:
        if "PtsRep" in args.model_name:
            rep_inf = rep_manage.RepManage(config, args)
            reps = rep_inf.get_reps(df)
        else:
            reps = rep_inf.generate_uni_reps(df)
    return reps


def main(args):
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    rep_inf = rep_manage.RepInference(config, args)
    
    if low_n_test_method: # 按照Low-N论文中的测试集选取方法，将总数据集分为三份，从其中一份中随机选取训练集（sub_train_df），将另两份数据集作为测试集（test_df）。
        sub_train_df, test_df = train_and_test.train_and_test_select_low_n(args, config)
    else: # 按照指定突变数目的方式选取测试集。根据 args.sampling_method 确定训练集的突变数目，根据参数 args.training_objectives 确定测试集的突变数目。
        sub_train_df, test_df = train_and_test.train_and_test_select(args, config)
    # sub_train_df.to_csv(f"/home/wangqihan/github/nn4dms-master/low-n_train_set/Low-N_train_set_{args.n_train_seqs}_seed_{args.seed}.csv")
    # print(dajfgbadjkgbhabgajbfgjklasdhasdjklh)

    if args.top_model_name != "inference": # 根据参数 args.top_model_name 确定下游模型。
        train_qfunc = np.array(sub_train_df["quantitative_function"])
        print(sub_train_df.head())
        print("train seq num: ", train_qfunc.shape)
        train_reps = generate_reps(args, sub_train_df)#蛋白质表征
        print('train reps:', train_reps[0])
        print('shape of train reps:', train_reps.shape)
        gener_top_model = modules.GenerateTopModel(config, args)

        if use_saved_model: # 直接使用已经保存好的下游模型。
            top_model = joblib.load(config.TOP_MODEL_PATH)
        else: # 使用训练数据训练下游模型。
            top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
            if save_top_model:
                top_model_saved_path = os.path.join(config.OUTPUT_PATH, "method_2", "top_model", args.top_model_name, f"top_model_{args.model_name}_{args.sampling_method}_train_num_{args.n_train_seqs}.pkl")
                joblib.dump(top_model, top_model_saved_path)
        
        if args.do_method3: # method3是在sub_train_df的基础上选择亮度高于0.6的突变株再训练一个回归模型，并测试其性能。
            print("---Do method3 ! Use the bright part of the training set(qfunc > 0.6) !---")
            step2_train_df = sub_train_df[sub_train_df["quantitative_function"] >= 0.6]
            step2_train_qfunc = np.array(step2_train_df["quantitative_function"])
            step2_train_reps = generate_reps(args, step2_train_df)
            step2_top_model = gener_top_model.sele_top_model(step2_train_reps, step2_train_qfunc)
    else:
        rep_inf = rep_manage.RepInference(config, args)


    print("do_test: ", args.do_test)
    if args.do_test == 'True': # 决定是否使用测试集测试模型性能，测试皮尔逊相关系数、斯皮尔曼相关系数、R^2 和 NDCG。
        print(test_df.head())
        print("test rep num: ", len(list(test_df["quantitative_function"])))
        if args.top_model_name == "inference":
            test_reps_inference = rep_inf.generate_inference(test_df["seq"])
            yhat = list(np.squeeze(test_reps_inference))
        else:
            test_reps = generate_reps(args, test_df)
            print("shape of test rep: ", test_reps.shape)
            yhat, yhat_std, yhat_mem  = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
            print(yhat[0:10])
        assert len(yhat) == len(list(test_df["quantitative_function"]))
        test_df[f"{args.model_name}_{args.top_model_name}"] = yhat
        results = train_and_test.result_statistics(args, config, test_df)
        if args.do_method3:
            if args.top_model_name != "inference":
                step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(test_reps, return_std=True, return_member_predictions=True)
                test_df[f"{args.model_name}_{args.top_model_name}_step2"] = step2_yhat
            result_dict = train_and_test.method3_result_statistics(args, config, test_df)
            print(result_dict)

    print("do_design: ", args.do_design)
    if args.do_design == "True": # 完成三个主要任务，第一：预测全部双突变的突变体；第二：根据双突变的突变体预测值计算双突变的位点组合；第三：根据双突变的位点组合计算三突变的位点组合。
        print("Do design method is :", design_method)
        input_dair_path = "/share/jake/hot_spot/data/all_2_mutation"
        ori_output_paht = os.path.join(config.OUTPUT_PATH, "method_2", "original_result", args.top_model_name, f"{args.model_name}_use_bright_{args.use_bright}_train_num_{args.n_train_seqs}_do_method3_{args.do_method3}_seed_{args.seed}")
        top_10_triple_mut_output_path = os.path.join(config.OUTPUT_PATH, "method_2", "top_10_3_mutation", args.top_model_name, args.model_name)
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
                all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_use_bright_{args.use_bright}_train_seqs_num_{args.n_train_seqs}_do_method3_{args.do_method3}_seed_{args.seed}_all_2_mut.csv"
                design_reps = generate_reps(args, df)
                
                design_yhat, yhat_std, yhat_mem  = top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                df[f"{args.model_name}_predict"] = design_yhat

                if args.do_method3:
                    design_step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                    df[f"{args.model_name}_predict_step2"] = design_step2_yhat
            df.to_csv(f"{ori_output_paht}/{file_name}")
            assert df.shape[0] == 361
            if args.do_method3:
                all_2_mutation_df = train_and_test.method2_satistic_step2(args, df, all_2_mutation_df)
            else:
                all_2_mutation_df = train_and_test.method2_satistic(args, df, all_2_mutation_df)
        all_2_mutation_df.to_csv(os.path.join(all_2_mut_output_path, all_2_mut_df_name)) # 保存双突变的位点组合的预测结果。

        all_3_mutation_df = pd.read_csv(os.path.join(config.DATA_SET_PATH ,"gfp_3_mutation_all.csv"))
        all_3_mut_output_path = os.path.join(config.OUTPUT_PATH, "method_2", "all_3_mutation", args.top_model_name)
        if args.top_model_name == "inference":
            all_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_seed_{args.seed}_all_3_mut.csv"
        else:
            all_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_use_bright_{args.use_bright}_train_seqs_num_{args.n_train_seqs}_seed_{args.seed}_all_3_mut.csv"
        all_3_mutation_df = stat_tri_mut.stati_triple_mut(all_2_mutation_df, all_3_mutation_df, args.stati_target)
        all_3_mutation_df.to_csv(os.path.join(all_3_mut_output_path, all_3_mut_df_name)) # 保存三突变的位点组合的预测结果。
        des_seqs_dir_name = f"{args.model_name}_use_bright_{args.use_bright}_train_num_{args.n_train_seqs}_seed_{args.seed}_{args.stati_target}"
        design_seqs_path = os.path.join(config.OUTPUT_PATH, "method_2", "design_seqs", args.top_model_name, des_seqs_dir_name)
        if not os.path.exists(design_seqs_path):
            os.mkdir(design_seqs_path)
        top_10_triple_mut = stat_tri_mut.design_seq_with_hotspot(all_3_mutation_df, args.stati_target, 10, design_seqs_path) # 根据三突变位点组合的预测结果，筛选出预测值最高的10个位点组合。并根据
        if not os.path.exists(top_10_triple_mut_output_path):
            os.mkdir(top_10_triple_mut_output_path)
        top_10_triple_mut.to_csv(os.path.join(top_10_triple_mut_output_path, f"top_10_{args.model_name}_{args.top_model_name}_{args.sampling_method}_use_bright_{args.use_bright}_train_num_{args.n_train_seqs}_seed_{args.seed}_{args.stati_target}_result.csv"))
    
    if args.predict_design_seqs:

        des_method = "from_Top20_mut_to_triple_mut_limit_aa"#new_method_top_100 lim_pos_top_10

        if design_method == "method_2":
            # classify_method_list = ["top_20%", "top_10%", "top_5%", "top_1%", "max"]
            classify_method_list = ["top_1%"]
            for cls_method in tqdm(classify_method_list):
                input_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs", args.top_model_name, f"{args.model_name}_use_bright_{args.use_bright}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}_do_method3_{args.do_method3}_des_method_{des_method}")
                output_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs_result", args.top_model_name, f"{args.model_name}_use_bright_{args.use_bright}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}_do_method3_{args.do_method3}_des_method_{des_method}")
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
                    print(f"{output_path}/{design_name}")
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
        "--stati_target", type=str, default="top_1%_func",
        help="""确定使用前百分之多少的双突变序列进行统计，共有（top_1%_func, top_5%_func, top_10%_func, top_20%_func）四种选择。"""
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