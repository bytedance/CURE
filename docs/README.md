
# CURE: Critical-Token-Guided Re-concatenation for Entropy-collapse Prevention
<p align="center">
  If you find this project useful, please give us a star 🌟.
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/BetuBin/CURE_Optimal_Model"><img src="https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface&logoColor=white" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k"><img src="https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface" alt="Hugging Face Dataset"></a>
  <a href="https://github.com/Bytedance-CURE/CURE/"><img src="https://img.shields.io/github/stars/Bytedance-CURE/CURE?style=social" alt="GitHub Repo stars"></a>
</p>

## 🔥 News
- *2024.8.14*: We’re excited to announce the release of CURE’s [paper](https://arxiv.org/abs/2508.11016).
- *2024.8.22*: We’re excited to announce the release of CURE’s [model](https://huggingface.co/bytedance-research/CURE).

## 📚 Algorithm Overview
![](https://github.com/Bytedance-CURE/CURE/blob/master/main.png) 

In Stage 1, given an input query 
$q$, the policy model produces a pool of candidate responses. We compute token-level entropy to identify critical tokens (high entropy), extract the clauses immediately preceding those tokens, append them to 
$q$ to form refined prompts, and query the model again. The newly generated responses are aggregated with the original ones and jointly optimized within a single group. In Stage 2, we continue training to translate the exploration bonus into realized performance.

## 📚 Experimental Results
![](https://github.com/Bytedance-CURE/CURE/blob/master/entropy.jpg)

Comparison of Entropy comparison of CURE-First-Stage and other methods at temperature 1.0.

![](https://github.com/Bytedance-CURE/CURE/blob/master/main_table.jpg)

CURE performs competitively compared with other algorithms. We report avg@32 for AIME24, AIME25, and AMC23 and avg@1 for others.


## ⚙️ Installation
Our code has been incorporated into VERL as a plugin, located in [recipe-CURE](https://github.com/Bytedance-CURE/CURE/tree/master/recipe)
### 1. Prepare the environment
Exactly the same as the environment setup in [verl](https://github.com/volcengine/verl), no additional configuration is required. In our actual workflow, we executed the following operations directly within the released image.
```bash
cd CURE
pip install --no-deps -e .
pip install --no-deps git+https://github.com/=hiyouga/MathRuler.git
pip install math_verify
```
### 2. Prepare the dataset and model
For training, simply download the dataset from [DAPO-Math-17K](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) and set its path as `TRAIN_FILE` in the [startup script](https://github.com/Bytedance-CURE/CURE/blob/master/recipe/dapo_cure/run_cure_stage_1.sh). We use [Qwen-2.5-Math-Base](https://huggingface.co/Qwen/Qwen2.5-Math-7B) as the training baseline; download it and set its path as `CKPTS_DIR` in the same script.
> We also recommend downloading [AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) and setting its path as `TEST_FILE` in the same script.


## 🚀 Train
### 1. First Stage Training
```bash
cd CURE
sh ./recipe/CURE_First_Stage/run_cure_stage_1.sh
```

### 2. Second Stage Training
```bash
cd CURE
sh ./recipe/CURE_Second_Stage/run_cure_stage_2.sh
```


## 💓 Acknowledgement
This project has been developed partially based on the following pioneering works on GitHub repositories.
We express our profound gratitude for these foundational resources:
- https://github.com/volcengine/verl
- https://github.com/NVlabs/NFT
- https://github.com/huggingface/Math-Verify

We would like to extend our special thanks to the following contributors [@Qingbin Li](https://github.com/BetuBin18070), [@Rongkun Xue](https://github.com/rongkunxue),  for their valuable contributions and support to this algorithm library.

## 🌏 Citation
```bibtex
@misc{li2025curecriticaltokenguidedreconcatenationentropycollapse,
      title={CURE: Critical-Token-Guided Re-concatenation for Entropy-collapse Prevention}, 
      author={Qingbin Li and Rongkun Xue and Jie Wang and Ming Zhou and Zhi Li and Xiaofeng Ji and Yongqi Wang and Miao Liu and Zheming Yang and Minghui Qiu and Jing Yang},
      year={2025},
      eprint={2508.11016},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.11016}, 
}
```

## 🏷️ License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
