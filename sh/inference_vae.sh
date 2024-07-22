THEANO_FLAGS='floatX=float32,device=cuda' python /home/wangqihan/Low-N-improvement/vae/vae_inference.py \
    --model_prefix GFP_AEQVI \
    --input_csv_path /share/jake/Low_N_data/csv \
    --fasta_file /home/wangqihan/github/combining-evolutionary-and-assay-labelled-data/data/GFP_AEQVI_Sarkisyan2016/seqs_focus_full.fasta \
    --input_csv_name  sk_data_set_distance\
    --output_dir /share/jake/Low_N_data/inference/vae