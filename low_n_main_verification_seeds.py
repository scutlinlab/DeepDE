import os
import numpy as np
import pandas as pd
import torch
import argparse
from scipy import stats
# from sklearn.metrics import r2_score

import config 
from low_n_utils import modules
from low_n_utils import rep_manage

from sklearn.metrics import r2_score, roc_auc_score, ndcg_score


def calc_ndcg(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / y_true.std() 
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def main(args):
    train_csv_path = f"/home/wangqihan/Low_N_test/cv_400_csv/set_train_{args.seed}.csv"
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    

    sele_low_n_df = modules.SelectLowNDataFram(config, args)
    print('Setting up training data')
    train_df = pd.read_csv(train_csv_path)
    sub_train_df = train_df#train_df.sample(n=args.n_train_seqs)

    train_qfunc = sele_low_n_df.get_qfunc(sub_train_df)

    print(sub_train_df.head())
    # print(sub_train_df['distance'])
    print(train_qfunc.shape)
    rep_mana = rep_manage.RepManage(config, args)
    train_reps = rep_mana.get_train_reps(sub_train_df)
    print('train_reps:', train_reps[0])
    print('train_reps:', train_reps.shape)
    # print(train_qfunc)
    # print(dksffasdfadsfas)
    gener_top_model = modules.GenerateTopModel(config, args)
    top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
    # train_info = {
    #             'top_model': top_model,
    #             'train_df': sub_train_df,
    #             'train_seq_reps': train_reps,
    #             'base_model': model_name
    #             }
    print(args.do_test)
    if args.do_test == 'True':
        # test_df = sele_low_n_df.select_test_df()
        test_df = pd.read_csv("/share/jake/Low_N_data/test_csv/sk_test_set_old.csv")
        print(list(test_df["seq"])[0])
        test_qfunc = sele_low_n_df.get_qfunc(test_df)
        test_reps = rep_mana.get_test_reps(test_df)
        print(test_reps.shape)
        yhat = rep_mana.predict_rep_fitness(top_model, test_reps)
        print(len(test_qfunc), len(yhat))
        print(test_qfunc[-96:])
        print(yhat[-96:])
        r = stats.pearsonr(test_qfunc, yhat)[0]
        rho = stats.spearmanr(test_qfunc, yhat).correlation
        r2 = r2_score(test_qfunc, yhat)
        ndcg = calc_ndcg(yhat, test_qfunc)
        pred_and_real = np.array(list(yhat) + list(test_qfunc))
        name_list = []
        for i in range(len(test_qfunc)):
            name_list.append(i)
        print("r : ", r, "rho : ", rho, "r2 : ", r2, "ndcg : ", ndcg)

        if args.save_test_result == 'True':
            rep_mana.save_result(pred_and_real, name_list, r, rho, r2, ndcg)
            if args.sampling_method == "random_1-3" or args.sampling_method == "random_1-2" :
                rep_mana.save_train_set_proportion(sub_train_df)
                
    if args.do_design == 'True':
        results = rep_mana.mcmc_design(top_model, sub_train_df, train_reps)
        print("Saving design results.")
        rep_mana.save_design_datas(results)

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