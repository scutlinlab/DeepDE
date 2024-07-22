
for s in $(seq 0 4)
do
python /home/wangqihan/Low-N-improvement/low_n_main_method3_classification.py \
    --seed $s \
    --gpu 2 \
    --sampling_method all \
    --n_train_seqs 1000 \
    --model_name onehot \
    --training_objectives gfp_SK_test_3 \
    --classfy_model SVM_rbf \
    --use_bright 0 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done
