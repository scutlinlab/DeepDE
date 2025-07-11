# DeepDE

DeepDE is a deep learning-driven framework for protein engineering that integrates supervised, unsupervised, and weak-positive-only learning strategies to guide mutagenesis and accelerate functional optimization. It is designed to address key challenges in the vast combinatorial space of protein sequence variation, especially for triple mutant exploration.

🚀 Key Features
This project implements a design strategy with a fixed mutation radius of 3,  expanding the search space to over 10¹⁰ possible variants. 
Two complementary design modes:
DM:  "mutagenesis by direct prediction" approach, which involves direct prediction of beneficial triple mutants with specific amino acid substitutions.
SM: "mutagenesis coupled with screening" approach, where potential beneficial triple mutation sites are predicted, followed by the experimental construction of 10 libraries of triple mutants for screening to identify the best mutants.
Iterative training and evolution cycles, mimicking directed evolution but enhanced by deep learning.
Experimental validation using avGFP as a model system, achieving up to 74.3-fold improvement in activity within 4 rounds, surpassing benchmarks like sfGFP.

🎯 Why DeepDE?
Classical directed evolution is labor-intensive and slow. Existing AI-guided approaches often lack iterative design capabilities and robust experimental validation. DeepDE overcomes both limitations by:
Incorporating data-efficient supervised learning (starting from only 1,000 single/double mutants).
Enabling generalization beyond training distribution.
Integrating mutational screening with model-based prediction in an iterative framework.

📈 Notable Results
Path III (SM-only strategy) showed the most consistent and effective optimization trajectory.
Outperformed state-of-the-art protein engineering algorithms like Low-N and ECNet.
Demonstrated capability to generalize and improve from limited data across multiple design cycles.


Inspired by classic directed protein evolution, we developed DeepDE, a protocol for deep learning-guided protein evolution. DeepDE incorporates three deep learning components (unsupervised, weak-positive only, and supervised learning) from the low-N algorithm. Key features of this approach include: 1) Expanding the supervised training dataset to 1,000 single or double mutants from the Sarkisyan dataset, using a 1/9 split setting. 2) Limiting the mutation radius to three, reducing computational workload and enabling the use of a standard mutagenesis kit to explore alternative triple mutants. 3) Implementing an iterative design cycle over five rounds.

Our work marks a significant advancement in protein design and engineering, with potential applications in biotechnology.Our model training was conducted on a workstation with dual Ubuntu 18.04.03 and NVIDIA GeForce RTX 2080Ti graphics-processing unit.

------

Software Name: DeepDE V1.0

Year: 2024

*Rights Statement*

```
All rights of the software and associated documentation are owned by Lin lab. This software is currently under application for software copyright registration in China with the application number 2023SR0456882. All rights and intellectual property of this software are protected under law.
```

*Usage and Distribution*

```
No one is permitted to use, copy, modify, merge, publish, distribute, sublicense, or deal in the software in any way, commercially or non-commercially, without explicit written permission.
```

*Contacts*

201910108050@mail.scut.edu.cn; 202120124523@mail.scut.edu.cn; biyangxf@scut.edu.cn; zhanglinlin@gdut.edu.cn



### 1. INSTALLATION

Standard (harder):

Requirements in Ubuntu:
  conda env create -f DeepDE.yml
  conda activate DeepDE

### 2. DeepDE USAGE
### Example Command (Design + Prediction)
```bash
python main.py \
  --seed 42 \
  --gpu 0 \
  --model_name eUniRep \
  --training_objectives gfp_SK_test_2 \
  --do_design True \
  --save_test_result False \
  --n_train_seqs 1000 \
  --sampling_method random \
  --top_model_name lin \
  --stati_target top_5%_func
### Predict Previously Designed Sequences
python main.py \
  --seed 42 \
  --gpu 0 \
  --model_name eUniRep \
  --training_objectives gfp_SK_test_2 \
  --do_design False \
  --predict_design_seqs 1 \
  --n_train_seqs 1000 \
  --sampling_method random \
  --top_model_name lin \
  --stati_target top_5%_func
## 3.Argument Explanation
| Argument | Description |
|----------|-------------|
| `--seed` | Random seed for reproducibility (**required**) |
| `--gpu` | GPU index to use (default: 0) |
| `--model_name` | Representation model to use (`UniRep`, `eUniRep`, `Random_UniRep`, `eUniRep-Augmenting`, etc.) |
| `--customize_train_set` | Path to a user-defined CSV training set (optional) |
| `--training_objectives` | Fitness objective label (e.g., `gfp_SK_test_2`) |
| `--do_design` | Whether to perform mutational design (`True` or `False`) |
| `--save_test_result` | Whether to save prediction results |
| `--n_train_seqs` | Number of sequences in training set |
| `--sampling_method` | How to sample training set (`random`, `bright`, etc.) |
| `--top_model_name` | Top model to use (`lin`, `rf`, `inference`) |
| `--use_bright` | Whether to use only bright sequences in training |
| `--stati_target` | Statistical filtering target for mutation site selection (`top_1%_func`, `top_5%_func`, etc.) |
| `--predict_design_seqs` | Whether to evaluate the designed sequences |

## 4.Output Structure
- `all_2_mutation/` – Prediction results of all 2-site mutants
- `all_3_mutation/` – Statistical rankings of triple mutations
- `design_seqs/` – Designed sequences selected by statistical filters
- `design_seqs_result/` – Predicted values of designed sequences

```
