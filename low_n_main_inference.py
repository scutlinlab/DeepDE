import os
import numpy as np
import pandas as pd
import torch
import argparse
from scipy import stats
# from sklearn.metrics import r2_score

import config 
from low_n_utils import models
from low_n_utils import modules
from low_n_utils import rep_manage
from low_n_utils import data_utils

from sklearn.metrics import r2_score, roc_auc_score, ndcg_score

predict_type = "Augmenting" #Augmenting
output_path = "/share/jake/Augmenting/test_result"

def calc_ndcg(y_pred, y_true):
    print(type(y_pred), type(y_true))
    y_true_normalized = (y_true - y_true.mean()) / y_true.std() 
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))

def main(args):
    save_data = {"seed": [], "pearsonr": [], "spearmanr": [], "r2": [], "ndcg": []}
    # for i in range(20):
    #     args.seed = i
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sele_low_n_df = modules.SelectLowNDataFram(config, args)
    rep_inf = rep_manage.RepInference(config, args)
    gener_top_model = modules.GenerateTopModel(config, args)
    if predict_type == "Augmenting":
        sub_train_df = sele_low_n_df.select_train_df()
        train_qfunc = sele_low_n_df.get_qfunc(sub_train_df)
        print(sub_train_df.head())
        # onehot_base_model = models.OneHotRegressionModel('EnsembledRidge')
        # train_reps_onehot = onehot_base_model.encode_seqs(list(sub_train_df["seq"]))
        train_reps_onehot = data_utils.seqs_to_onehot(list(sub_train_df["seq"]))
        train_reps_inference = - rep_inf.generate_uni_inference(sub_train_df["seq"])
        print(train_reps_onehot.shape, train_reps_inference.shape)
        train_reps = np.concatenate([train_reps_onehot, train_reps_inference], axis=1)
        top_model = gener_top_model.sele_top_model(train_reps, train_qfunc)
        print(train_reps.shape)
        # print(dfahjfajkfgahfjkasdhdfhjkhsdfjghjklsd)
        print(args.do_test)
    if args.do_test == 'True':
        test_df = sele_low_n_df.select_test_df()
        print(list(test_df["seq"])[0])
        test_qfunc = sele_low_n_df.get_qfunc(test_df)
        test_reps_inference = - rep_inf.generate_uni_inference(test_df["seq"])
        if predict_type == "inference":
            yhat = list(np.squeeze(test_reps_inference))
            test_df[f"{args.model_name}_inference"] = yhat
            # print(np.squeeze(test_reps_inference).shape)
        elif predict_type == "Augmenting":
            # test_reps_onehot = onehot_base_model.encode_seqs(list(test_df["seq"]))
            test_reps_onehot = data_utils.seqs_to_onehot(list(test_df["seq"]))
            test_reps = np.concatenate([test_reps_onehot, test_reps_inference], axis=1)
            yhat, yhat_std, yhat_mem  = top_model.predict(test_reps, return_std=True, return_member_predictions=True)
            test_df[f"{args.model_name}_Augmenting"] = yhat
        print(len(test_qfunc), len(yhat))
        # print(len(yhat[0]))
        
        save_data["seed"].append(args.seed)
        r = stats.pearsonr(test_qfunc, yhat)[0]
        save_data["pearsonr"].append(r)
        rho = stats.spearmanr(test_qfunc, yhat).correlation
        save_data["spearmanr"].append(rho)
        r2 = r2_score(test_qfunc, yhat)
        save_data["r2"].append(r2)
        yhat_ndcg = np.array(yhat)
        ndcg = calc_ndcg(yhat_ndcg, test_qfunc)
        save_data["ndcg"].append(ndcg)
        print(save_data)
        pred_and_real = np.array(list(yhat) + list(test_qfunc))
        name_list = []
        for i in range(len(test_qfunc)):
            name_list.append(i)
        print("r : ", r, "rho : ", rho, "r2 : ", r2, "ndcg : ", ndcg)

    if args.save_test_result == 'True':
        if predict_type == "inference":
            test_df.to_csv(f"/share/jake/Augmenting/test_result/{args.model_name}/predict_qfunc/{args.model_name}_{predict_type}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_seed_{args.seed}.csv")
        elif predict_type == "Augmenting":
            test_df.to_csv(f"/share/jake/Augmenting/test_result/{args.model_name}/predict_qfunc/{args.model_name}_{predict_type}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}_seed_{args.seed}.csv")
        # save_df = pd.DataFrame(save_data)
        # print(save_df)
        # save_df.to_csv(f"/share/jake/Augmenting/test_result/{args.model_name}/{args.model_name}_{args.sampling_method}_{args.training_objectives}_n_train_seqs_{args.n_train_seqs}.csv")
                
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