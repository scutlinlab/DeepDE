import os
import numpy as np
import pandas as pd
import argparse

import config 
from network import statistics_triple_mutation as stat_tri_mut
from network import rep_manage
from network import train_and_test


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
    if args.customize_train_set is not None:
        print("Use customize train set! Load train set from : ", args.customize_train_set)
        sub_train_df = pd.read_csv(args.customize_train_set)
    else:
        print("Use Sarkisyan train set!")
        sub_train_df, test_df = train_and_test.train_and_test_select(args, config)
    
    if args.top_model_name != "inference":
        train_qfunc = np.array(sub_train_df["quantitative_function"])
        print(sub_train_df.head())
        print("train seq num: ", train_qfunc.shape)
        train_reps = generate_reps(args, sub_train_df)
        print('train reps:', train_reps[0])
        print('shape of train reps:', train_reps.shape)
        gener_top_model = rep_manage.GenerateTopModel(config, args)
        top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
    else:
        rep_inf = rep_manage.RepInference(config, args)

    print("do_design: ", args.do_design)
    if args.do_design == "True":
        
        all_2_mut_output_path = os.path.join(config.OUTPUT_PATH, "all_2_mutation", args.top_model_name)
        top_10_triple_mut_output_path = os.path.join(config.OUTPUT_PATH, "top_10_3_mutation", args.top_model_name, args.model_name)
        all_2_mutation_df = pd.read_csv(os.path.join(config.DATA_SET_PATH ,"gfp_2_mutation_all.csv"))
        all_2_mutation_seq_list = all_2_mutation_df["seq"].to_list()
        all_2_mutation_name_list = all_2_mutation_df["name"].to_list()
        if args.top_model_name == "inference":
            all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_seed_{args.seed}_all_2_mut.csv"
        else:
            if args.customize_train_set is not None:
                customize_name = args.customize_train_set.split("/")[-1].split(".")[0]
                all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_customize_train_set_{customize_name}_seed_{args.seed}_all_2_mut.csv"
            else:
                all_2_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_seqs_num_{args.n_train_seqs}_seed_{args.seed}_all_2_mut.csv"
        
        # for i, file_name in enumerate(all_file_name):
        for i, mut_seq_name in enumerate(all_2_mutation_name_list):
            temp_seq = all_2_mutation_seq_list[i]
            df = stat_tri_mut.generate_seq_from_mut_site(temp_seq, mut_seq_name)
           
            if args.top_model_name == "inference":
                design_reps_inference = rep_inf.generate_inference(df["seq"])
                design_yhat = list(np.squeeze(design_reps_inference))
                df[f"{args.model_name}_inference"] = design_yhat
            else:
                design_reps = generate_reps(args, df)
                design_yhat, yhat_std, yhat_mem  = top_model.predict(design_reps, return_std=True, return_member_predictions=True)
                df[f"{args.model_name}_predict"] = design_yhat

            assert df.shape[0] == 361
            all_2_mutation_df = train_and_test.method2_satistic(args, df, all_2_mutation_df)

        all_2_mutation_df.to_csv(os.path.join(all_2_mut_output_path, all_2_mut_df_name))
        all_3_mutation_df = pd.read_csv(os.path.join(config.DATA_SET_PATH ,"gfp_3_mutation_all.csv"))
        all_3_mut_output_path = os.path.join(config.OUTPUT_PATH, "all_3_mutation", args.top_model_name)
        if args.top_model_name == "inference":
            all_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_seed_{args.seed}_all_3_mut.csv"
        else:
            if args.customize_train_set is not None:
                all_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_customize_train_set_{customize_name}_seed_{args.seed}_all_3_mut.csv"
            else:
                all_3_mut_df_name = f"{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_seqs_num_{args.n_train_seqs}_seed_{args.seed}_all_3_mut.csv"
        all_3_mutation_df = stat_tri_mut.stati_triple_mut(all_2_mutation_df, all_3_mutation_df, args.stati_target)
        all_3_mutation_df.to_csv(os.path.join(all_3_mut_output_path, all_3_mut_df_name))
        des_seqs_dir_name = f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{args.stati_target}"
        design_seqs_path = os.path.join(config.OUTPUT_PATH, "design_seqs", args.top_model_name, des_seqs_dir_name)
        if not os.path.exists(design_seqs_path):
            os.mkdir(design_seqs_path)
        top_10_triple_mut = stat_tri_mut.design_seq_with_hotspot(all_3_mutation_df, args.stati_target, 10, design_seqs_path)
        if not os.path.exists(top_10_triple_mut_output_path):
            os.mkdir(top_10_triple_mut_output_path)
        top_10_triple_mut.to_csv(os.path.join(top_10_triple_mut_output_path, f"top_10_{args.model_name}_{args.top_model_name}_{args.sampling_method}_train_num_{args.n_train_seqs}_seed_{args.seed}_{args.stati_target}_result.csv"))


    if args.predict_design_seqs:
        input_path = design_seqs_path
        
        if args.customize_train_set is not None:
            output_path = os.path.join(config.OUTPUT_PATH, "design_seqs_result", args.top_model_name, f"{args.model_name}_customize_train_set_{customize_name}_seed_{args.seed}_{args.stati_target}")
        else:
            output_path = os.path.join(config.OUTPUT_PATH, "design_seqs_result", args.top_model_name, f"{args.model_name}_train_num_{args.n_train_seqs}_seed_{args.seed}_{args.stati_target}")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        all_design_names = os.listdir(input_path)
        for design_name in all_design_names:
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
            
            print(f"{output_path}/{design_name}")
            des_df.to_csv(f"{output_path}/{design_name}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int,
        help="""指定随机数种子."""
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="""可以指定使用服务器上的特定GPU."""
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="""可以选择使用哪种表征方法进行辅助定向进化,现有的表征方法有 -UniRep -eUniRep -Random_UniRep -eUniRep-Augmenting等."""
    )
    parser.add_argument(
        "--customize_train_set", type=str, default=None,
        help="""是否自定义数据集."""
    )
    parser.add_argument(
        "--training_objectives", type=str, default="gfp_SK_test_2",
        help="""可以选择特定的测试目标"""
    )
    parser.add_argument(
        "--do_design", type=str, default='True',
        help="""可以选择是否进行蛋白质的设计步骤""",
        required=True
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
        "--top_model_name", type=str, default="lin",
        help="""确定使用何种下游模型默认为线性回归"""
    )
    parser.add_argument(
        "--use_bright", type=int, default=0,
        help="""是否只是使用亮的序列作为训练集"""
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
    if(args.save_test_result is None):
        raise ValueError(
        "--save_test_result must be specified"
                            )
    main(args)