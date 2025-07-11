
for s in $(seq 0 20)
do
python /home/wangqihan/Low-N-improvement/low_n_main_inference_new.py \
    --seed $s \
    --gpu 1 \
    --sampling_method random_1 \
    --n_train_seqs 400 \
    --model_name eUniRep \
    --training_objectives gfp_SK \
    --use_bright 0 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done
