python design.py -c configs/sa_config_gfp_unirep.json -m unirep -n 8 -e 21092902 -g 0&
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_1 -n 8 -e 21092903 -g 6&
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_2 -n 8 -e 21092904 -g 7&
python design.py -c configs/sa_config_gfp_unirep.json -m random_unirep -n 8 -e 21092905 -g 2&
wait
python design.py -c configs/sa_config_gfp_unirep.json -m onehot -n 8 -e 21092906 -g 0&
python design.py -c configs/sa_config_gfp_unirep.json -m unirep -n 24 -e 21092907 -g 7&
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_1 -n 24 -e 21092908 -g 2&
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_2 -n 24 -e 21092909 -g 6&
wait
python design.py -c configs/sa_config_gfp_unirep.json -m random_unirep -n 24 -e 21092910 -g 7&
python design.py -c configs/sa_config_gfp_unirep.json -m onehot -n 24 -e 21092911 -g 6&
python design.py -c configs/sa_config_gfp_unirep.json -m unirep -n 96 -e 21092912 -g 0&
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_1 -n 96 -e 21092913 -g 2&
wait
python design.py -c configs/sa_config_gfp_unirep.json -m eunirep_2 -n 96 -e 21092914 -g 0&
python design.py -c configs/sa_config_gfp_unirep.json -m random_unirep -n 96 -e 21092915 -g 6&
python design.py -c configs/sa_config_gfp_unirep.json -m onehot -n 96 -e 21092916 -g 2&

python design.py -c configs/sa_config_gfp.json -n 8 -e 21092917 -g 0&
python design.py -c configs/sa_config_gfp.json -n 24 -e 21092918 -g 0&
python design.py -c configs/sa_config_gfp.json -n 96 -e 21092919 -g 0&
python design.py -c configs/sa_config_gfp.json -n 400 -e 21092920 -g 0&