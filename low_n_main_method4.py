import os
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score

import config 
from low_n_utils import modules
from low_n_utils import rep_manage

choose_step = "step2"
classfy_score = 0.6
step2_bright_degree = 0.6

def main(args):
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # torch.set_num_threads(1)#限制CPU占用量
    # device_num = 0
    # torch.cuda.set_device(device_num)
    # device = torch.device('cuda:%d' % device_num)
    # if torch.cuda.is_available():
    #     print('GPU available!!!')
    #     print('MainDevice=', device)

    sele_low_n_df = modules.SelectLowNDataFram(config, args)
    # sub_train_df = sele_low_n_df.select_train_df()
    assert args.training_objectives == "gfp_SK_test_3"
    if choose_step == "step1":
        assert args.use_bright
        print("use_bright_train_set!")
        sub_train_df_all = pd.read_csv(f"/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_{args.seed}.csv")
        sub_train_df_all.drop(sub_train_df_all.columns[[0]], axis=1,inplace=True)
        sub_train_df = sub_train_df_all[sub_train_df_all["quantitative_function"] >= classfy_score]
        print("step1 train_num: ", len(list(sub_train_df["quantitative_function"])))
    elif choose_step == "step2":
        print("use_normal_train_set!")
        sub_train_df = pd.read_csv(f"/share/jake/Low_N_data/train_csv/gfp_SK_test_3_train_random_1-2_1000_{args.seed}.csv")
        sub_train_df.drop(sub_train_df.columns[[0]], axis=1,inplace=True)
        print("step2 train_num: ", len(list(sub_train_df["quantitative_function"])))
    # sub_train_df.to_csv(f"/share/jake/Low_N_data/train_csv/{args.training_objectives}_{args.model_name}_train_{args.sampling_method}_1000_{args.seed}.csv")
    train_qfunc = sele_low_n_df.get_qfunc(sub_train_df)

    print(sub_train_df.head())
    # print(sub_train_df['distance'])
    print("train rep ebd shape: " ,train_qfunc.shape)
    rep_mana = rep_manage.RepManage(config, args)
    train_reps = rep_mana.get_train_reps(sub_train_df)
    print(train_reps[0])
    print('train_reps:', train_reps.shape)
    # print(train_qfunc)

    gener_top_model = modules.GenerateTopModel(config, args)
    top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
    print("If do test ? :", args.do_test)

    if args.do_test == 'True':
        if choose_step == "step2":
            if classfy_score == 1:
                test_df = pd.read_csv(f"/share/jake/hot_spot/data/test/method4/step1/test_set_for_step2/{args.model_name}_calssfy_degree{step2_bright_degree}_test_{choose_step}_vary_bright_{args.seed}.csv")
            elif classfy_score == 0.6:
                test_df = pd.read_csv(f"/share/jake/hot_spot/data/test/method4/step1/test_set_for_step2/{args.model_name}_calssfy_degree{step2_bright_degree}_test_{choose_step}_bright_{args.seed}.csv")
            # test_df.drop(test_df.columns[[0]], axis=1,inplace=True)
        else:
            test_df = sele_low_n_df.select_test_df()
        test_qfunc = sele_low_n_df.get_qfunc(test_df)
        test_reps = rep_mana.get_test_reps(test_df)
        print("test rep num: " ,len(test_qfunc))
        print("test rep ebd shape: " ,test_reps.shape)
        yhat = rep_mana.predict_rep_fitness(top_model, test_reps)
        print("fsgsdfdgsfgsdfgsfgsf")
        print(test_df.head())
        assert len(test_qfunc) == len(yhat)
        if choose_step == "step1":
            test_df["predict_qfunc"] = list(yhat)
            if classfy_score == 1:
                test_df.to_csv(f"/share/jake/hot_spot/data/test/method4/{choose_step}/result_csv/{args.model_name}_1-2muta_train_3muta_predict_{args.n_train_seqs}_{choose_step}_vary_bright_{args.seed}.csv")
            elif classfy_score == 0.6:
                test_df.to_csv(f"/share/jake/hot_spot/data/test/method4/{choose_step}/result_csv/{args.model_name}_1-2muta_train_3muta_predict_{args.n_train_seqs}_{choose_step}_bright_{args.seed}.csv")
        else:
            test_df["predict_qfunc_step2"] = list(yhat)
            if classfy_score == 1:
                test_df.to_csv(f"/share/jake/hot_spot/data/test/method4/{choose_step}/result_csv/{args.model_name}_calssfy_degree{step2_bright_degree}_result_{choose_step}_vary_bright_{args.seed}.csv")
            elif classfy_score == 0.6:
                test_df.to_csv(f"/share/jake/hot_spot/data/test/method4/{choose_step}/result_csv/{args.model_name}_calssfy_degree{step2_bright_degree}_result_{choose_step}_bright_{args.seed}.csv")
            
        # print(test_qfunc[-96:])
        # print(yhat[-96:])
        r = stats.pearsonr(test_qfunc, yhat)[0]
        rho = stats.spearmanr(test_qfunc, yhat).correlation
        r2 = r2_score(test_qfunc, yhat)
        pred_and_real = np.array(list(yhat) + list(test_qfunc))
        name_list = []
        for i in range(len(test_qfunc)):
            name_list.append(i)
        print("r : ", r, "rho : ", rho, "r2 : ", r2)

        if args.save_test_result == 'True':
            rep_mana.save_result(pred_and_real, name_list, r, rho, r2)
            if args.sampling_method == "random_1-3" or args.sampling_method == "random_1-2" :
                rep_mana.save_train_set_proportion(sub_train_df)
                
    if args.do_design == 'True':
        assert args.use_bright
        assert "step2" in args.training_objectives
        seed = 0
        method = "method1"
        train_seq_num = "all"
        objective = "max"
        assert seed == args.seed
        if method == "method2":
            input_dair_path = f"/share/jake/hot_spot/data/result/method_2/design_seqs_result/all_3_mutation_1-2_{objective}_result_{train_seq_num}_{seed}_new"
            output_dair_path = f"/share/jake/hot_spot/data/result/method_3/design_seqs_result/{method}/{args.model_name}/all_3_mutation_1-2_{objective}_result_{train_seq_num}_{seed}"
        elif method == "method1":
            input_dair_path = f"/share/jake/hot_spot/data/result/method_1/design_seqs_result/all_3_mutation_{objective}_result_{train_seq_num}_{seed}_new"
            output_dair_path = f"/share/jake/hot_spot/data/result/method_3/design_seqs_result/{method}/{args.model_name}/all_3_mutation_{objective}_result_{train_seq_num}_{seed}"
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