for s in $(seq 0 4)
do
python /home/wangqihan/Low-N-improvement/low_n_main.py \
    --seed $s \
    --gpu 1 \
    --sampling_method random \
    --n_train_seqs 400 \
    --model_name UniRep \
    --training_objectives gfp_SK_low_predict_hight_split \
    --use_bright 1 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done&
for s in $(seq 0 4)
do
python /home/wangqihan/Low-N-improvement/low_n_main.py \
    --seed $s \
    --gpu 2 \
    --sampling_method random \
    --n_train_seqs 400 \
    --model_name UniRep \
    --training_objectives gfp_SK_low_predict_hight_split \
    --use_bright 1 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done&
for s in $(seq 0 4)
do
python /home/wangqihan/Low-N-improvement/low_n_main.py \
    --seed $s \
    --gpu 6 \
    --sampling_method random \
    --n_train_seqs 400 \
    --model_name UniRep \
    --training_objectives gfp_SK_low_predict_hight \
    --use_bright 1 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done&
for s in $(seq 0 4)
do
python /home/wangqihan/Low-N-improvement/low_n_main.py \
    --seed $s \
    --gpu 7 \
    --sampling_method random \
    --n_train_seqs 400 \
    --model_name UniRep \
    --training_objectives gfp_SK_low_predict_hights \
    --use_bright 1 \
    --do_design Fales \
    --do_test True \
    --save_test_result True
done&

