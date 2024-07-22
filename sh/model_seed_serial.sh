
for s in $(seq 0 4)
do
python /home/wangqihan/github/Low-N-improvement/low_n_main.py \
    --seed $s \
    --gpu 2 \
    --sampling_method random_1-2 \
    --n_train_seqs 96 \
    --top_model_name inference \
    --model_name eUniRep \
    --training_objectives gfp_SK_split_test_2 \
    --use_bright 0 \
    --do_method3 0 \
    --predict_design_seqs 0 \
    --do_design False \
    --do_test True \
    --save_test_result True
done
