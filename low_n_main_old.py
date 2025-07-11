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
from low_n_utils import data_utils
from low_n_utils import train_and_test

from sklearn.metrics import r2_score, roc_auc_score, ndcg_score

low_n_test_method = False
design_method = "method_2"


def main(args):
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    rep_inf = rep_manage.RepInference(config, args)
    if low_n_test_method:
        sub_train_df, test_df = train_and_test.train_and_test_select_low_n(args, config)
    else:
        sub_train_df, test_df = train_and_test.train_and_test_select(args, config)
    if args.do_design == "True" and design_method == "method_1":
        assert args.top_model_name != "inference"#The model does not support method 1
        assert "Augmenting" not in args.model_name#The model does not support method 1
        sub_train_df = pd.read_csv("/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_0_hotspot.csv")
        train_qfunc = np.array(sub_train_df["max_quantitative_function"])
        train_reps = rep_inf.generate_uni_reps(sub_train_df, hotspot = True)
        print(train_qfunc)
        print('train reps:', train_reps[0])
        print('shape of train reps:', train_reps.shape)
        gener_top_model = modules.GenerateTopModel(config, args)
        top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)

    else:
        if low_n_test_method:
            sub_train_df, test_df = train_and_test.train_and_test_select_low_n(args, config)
        else:
            sub_train_df, test_df = train_and_test.train_and_test_select(args, config)
        # '''-----------------------------debug--------------------------------'''
        # sub_train_df = pd.read_csv("/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_0.csv")
        # '''------------------------------------------------------------------'''

        if args.top_model_name != "inference":
            train_qfunc = np.array(sub_train_df["quantitative_function"])
            print(sub_train_df.head())
            print("train seq num: ", train_qfunc.shape)
            if "Augmenting" in args.model_name:
                if "concate" in args.model_name:
                    train_reps = rep_inf.concate(sub_train_df)
                else:
                    train_reps = rep_inf.Augmenting(sub_train_df)
            else:
                train_reps = rep_inf.generate_uni_reps(sub_train_df)
            print('train reps:', train_reps[0])
            print('shape of train reps:', train_reps.shape)
            gener_top_model = modules.GenerateTopModel(config, args)
            top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
            
        else:
            rep_inf = rep_manage.RepInference(config, args)
    print("do_test: ", args.do_test)
    if args.do_test == 'True':
        print(test_df.head())
        print("test rep num: ", len(list(test_df["quantitative_function"])))
        if args.top_model_name == "inference":
            test_reps_inference = rep_inf.generate_inference(test_df["seq"])
            yhat = list(np.squeeze(test_reps_inference))
        else:
            if "Augmenting" in args.model_name:
                if "concate" in args.model_name:
                    test_reps = rep_inf.concate(test_df)
                else:
                    test_reps = rep_inf.Augmenting(test_df)
            else:
                test_reps = rep_inf.generate_uni_reps(test_df)
            print("shape of test rep: ", test_reps.shape)
            yhat, yhat_std, yhat_mem  = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
            print(yhat[0:10])
        assert len(yhat) == len(list(test_df["quantitative_function"]))
        test_df[f"{args.model_name}_{args.top_model_name}"] = yhat
        results = train_and_test.result_statistics(args, config, test_df)   

                
    if args.do_method3:
        print(test_df.head())
        print("Step 1 test rep num: ", len(list(test_df["quantitative_function"])))
        if args.top_model_name == "inference":
            print("Do method 3 inference ! Only step 1 ! training objectives: ", args.training_objectives)
            test_reps_inference = rep_inf.generate_inference(test_df["seq"])
            yhat = list(np.squeeze(test_reps_inference))
            assert len(yhat) == len(list(test_df["quantitative_function"]))
            test_df[f"{args.model_name}_{args.top_model_name}_step1"] = yhat
        # assert args.top_model_name != "inference"
        else:
            print("Do method 3 ,training objectives: ", args.training_objectives, "train seq sampling method: ", args.sampling_method)
            step1_top_model = top_model
            if "Augmenting" in args.model_name:
                if "concate" in args.model_name:
                    step1_test_reps = rep_inf.concate(test_df)
                else:
                    step1_test_reps = rep_inf.Augmenting(test_df)
            else:
                step1_test_reps = rep_inf.generate_uni_reps(test_df)
            step1_yhat, yhat_std, yhat_mem  = step1_top_model.predict(step1_test_reps, return_std=True, return_member_predictions=True)
            assert len(step1_yhat) == len(list(test_df["quantitative_function"]))
            test_df[f"{args.model_name}_{args.top_model_name}_step1"] = step1_yhat
            # step2_test_df = train_and_test.data_processing_method3_step1(args, test_df)
            step2_train_df = sub_train_df[sub_train_df["quantitative_function"] >= 0.6]
            step2_train_qfunc = np.array(step2_train_df["quantitative_function"])
            print("step2 train num: ", step2_train_qfunc.shape)
            if "Augmenting" in args.model_name:
                if "concate" in args.model_name:
                    step2_train_reps = rep_inf.concate(step2_train_df)
                else:
                    step2_train_reps = rep_inf.Augmenting(step2_train_df)
            else:
                step2_train_reps = rep_inf.generate_uni_reps(step2_train_df)
            step2_top_model = gener_top_model.sele_top_model(step2_train_reps, step2_train_qfunc)
            if "Augmenting" in args.model_name:
                if "concate" in args.model_name:
                    step2_test_reps = rep_inf.concate(test_df)
                else:
                    step2_test_reps = rep_inf.Augmenting(test_df)
            else:
                step2_test_reps = rep_inf.generate_uni_reps(test_df)
            step2_yhat, yhat_std, yhat_mem  = step2_top_model.predict(step2_test_reps, return_std=True, return_member_predictions=True)
            test_df[f"{args.model_name}_{args.top_model_name}_step2"] = step2_yhat
        result_dict = train_and_test.method3_result_statistics(args, config, test_df)
        print(result_dict)

    if args.do_design == "True":
        print("Do design method is :", design_method)
        if design_method == "method_1":
            input_dair_path = "/share/jake/hot_spot/data/all_3_mutation"
            three_mutation_ebd_path = "/share/jake/Low_N_data/ebd/eUniRep/GFP_3mutation_all"
            output_dair_path = os.path.join(config.OUTPUT_PATH, "method_1", "predict_qfunc", args.top_model_name, args.model_name)
            # output_dair_path = "/share/jake/github/low_n_output/method_1/predict_qfunc/lin/temp/new"
            if not os.path.exists(output_dair_path):
                os.mkdir(output_dair_path)
            all_file_name = os.listdir(input_dair_path)#debug
            for file_name in tqdm(all_file_name):
                input_path = f"{input_dair_path}/{file_name}"
                df = pd.read_csv(input_path)
                df.drop(df.columns[[0]], axis=1,inplace=True)
                rep_mana = rep_manage.RepManage(config, args)
                rep_array = rep_mana.load_rep_by_name(three_mutation_ebd_path ,df["name"])
                yhat, yhat_std, yhat_mem  = top_model.predict(rep_array, return_std=True, return_member_predictions=True)
                df[f"{args.model_name}_predict_qfunce"] = list(yhat)
                df.to_csv(f"{output_dair_path}/{file_name}")
        
        elif design_method == "method_2":
            
            input_dair_path = "/share/jake/hot_spot/data/all_2_mutation"
            ori_output_paht = os.path.join(config.OUTPUT_PATH, "method_2", "original_result", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}")
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
                    all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_seed_{args.seed}_all_2_mut.csv"
                    design_reps_inference = rep_inf.generate_inference(df["seq"])
                    design_yhat = list(np.squeeze(design_reps_inference))
                    df[f"{args.model_name}_inference"] = design_yhat
                else:
                    all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_seqs_num_{args.n_train_seqs}_seed_{args.seed}_all_2_mut.csv"
                    # top_1000_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_seqs_num_{args.n_train_seqs}_seed_{args.seed}_top_1000_3_mut.csv"
                    # assert args.n_train_seqs == 1000
                    if "Augmenting" in args.model_name:
                        if "concate" in args.model_name:
                            design_reps = rep_inf.concate(df)
                        else:
                            design_reps = rep_inf.Augmenting(df)
                    else:
                        design_reps = rep_inf.generate_uni_reps(df)
                    design_yhat, yhat_std, yhat_mem  = top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                    df[f"{args.model_name}_predict"] = design_yhat
                df.to_csv(f"{ori_output_paht}/{file_name}")
                assert df.shape[0] == 361
                all_2_mutation_df = train_and_test.method2_satistic(args, df, all_2_mutation_df)
            all_2_mutation_df.to_csv(os.path.join(all_2_mut_output_path, all_2_mut_df_name))
    
    if args.predict_design_seqs:
        if design_method == "method_2":
            classify_method_list = ["top_20%", "top_10%", "top_5%", "top_1%", "max"]
            for cls_method in tqdm(classify_method_list):
                input_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}")
                output_path = os.path.join(config.OUTPUT_PATH, design_method, "design_seqs_result", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{cls_method}")
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                all_design_names = os.listdir(input_path)
                for design_name in all_design_names:
                    pos_name = design_name.split(".")[0]
                    des_df = pd.read_csv(f"{input_path}/{design_name}")
                    if args.top_model_name == "inference":
                        des_reps_inference = rep_inf.generate_inference(des_df["seq"])
                        yhat = list(np.squeeze(des_reps_inference))
                    else:
                        if "Augmenting" in args.model_name:
                            if "concate" in args.model_name:
                                des_reps = rep_inf.concate(des_df)
                            else:
                                des_reps = rep_inf.Augmenting(des_df)
                        else:
                            des_reps = rep_inf.generate_uni_reps(des_df)
                        print("shape of test rep: ", des_reps.shape)
                        yhat, yhat_std, yhat_mem  = top_model.predict(des_reps, return_std=True, return_member_predictions=True)
                        print(yhat[0:10])
                    des_df[f"{args.model_name}_{args.top_model_name}_predict"] = yhat
                    des_df = train_and_test.compare_des_seqs(args, des_df)
                    des_df = des_df[(des_df["mutation_pos"] == pos_name)]
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