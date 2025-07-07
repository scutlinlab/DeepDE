# DeepDE

DeepDE is a deep learning-driven framework for protein engineering that integrates supervised, unsupervised, and weak-positive-only learning strategies to guide mutagenesis and accelerate functional optimization. It is designed to address key challenges in the vast combinatorial space of protein sequence variation, especially for triple mutant exploration.
🚀 Key Features
Triple-mutation design with a fixed mutation radius of 3, expanding the search space to over 10¹⁰ variants.
Two complementary design modes:
DM (Direct Prediction): Direct inference of beneficial triple mutants.
SM (Screening-guided Mutagenesis): Predictive identification of triple mutation sites, followed by targeted experimental library construction.
Iterative training and evolution cycles, mimicking directed evolution but enhanced by machine learning.
Experimental validation using avGFP as a model system, achieving up to 74.3-fold improvement in activity within 4 rounds, surpassing benchmarks like sfGFP.
🎯 Why DeepDE?
Classical directed evolution is labor-intensive and slow. Existing AI-guided approaches often lack iterative design capabilities and robust experimental validation. DeepDE overcomes both limitations by:
Incorporating data-efficient supervised learning (starting from only 1,000 single/double mutants).
Enabling generalization beyond training distribution.
Integrating mutational screening with model-based prediction in an iterative framework.
📈 Notable Results
Path III (SM-only strategy) showed the most consistent and effective optimization trajectory.
Outperformed state-of-the-art protein engineering algorithms like Low-N and EVOLVEpro.
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

### 2. DeepDE USAGE

```
# running on command line
source activate DeepDE
# change the main paraments in ./sh/model_seed_single.sh
python "the path to low_n_mian.py"
--seed the random seed to choose the training dataset, default 0
--gpu the gpu id for the script used
--use_bright 1 for only used the data which score >=1.04 for training, and 0 do not have any limitations
--predict_design_seqs 1 for used the trained model to design the mutation, 0 for nothing
--do_design True for training the DeepDE model amd 0 for nothing
```
```
⚠ Note: Due to server-related issues, the latest version of the code will be available on July 10th.
