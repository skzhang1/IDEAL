# IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models

<div style="text-align: center;">
    <img src="assets/ideal-logo.png" width=100% >
</div>

[Shaokun Zhang](https://github.com/skzhang1)<sup>1*</sup>, [Xiaobo Xia](https://xiaoboxia.github.io)<sup>2*</sup>, [Zhaoqing Wang](https://derrickwang005.github.io)<sup>2</sup>, [Ling-Hao Chen](https://lhchen.top)<sup>3</sup>, [Jiale Liu](https://leoljl.github.io)<sup>4</sup>, [Qingyun Wu](https://qingyun-wu.github.io)<sup>1</sup>, [Tongliang Liu](https://tongliang-liu.github.io)<sup>2</sup>

<sup>1</sup>Pennsylvania State University, <sup>2</sup>The University of Sydney, <sup>3</sup>Tsinghua University, <sup>4</sup>Xidian University

<sup>*</sup>Equal Contribution.
<p align="center">
  <a href='https://arxiv.org/abs/2310.10873'>
    <img src='https://img.shields.io/badge/Arxiv-2310.10873-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2310.10873.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://skzhang1.github.io/IDEAL/'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://github.com/skzhang1/IDEAL'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href='https://arxiv.org/abs/2310.10873'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg'>
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=skzhang1.IDEAL&left_color=gray&right_color=orange">
  </a>
</p>

**Official implementation for paper [IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models](https://arxiv.org/abs/2310.10873).**

## 🏠 Abstract

In-context learning is a promising paradigm that utilizes In-context examples as prompts for the predictions of large language models. These prompts are crucial for achieving strong performance. However, since the prompts need to be sampled from a large volume of annotated examples, finding the right prompt may result in high annotation costs. To address this challenge, this paper introduces an influence-driven selective annotation method that aims to minimize annotation costs while improving the quality of In-context examples. The essence of our method is
to select a pivotal subset from a large-scale unlabeled data pool to annotate for the subsequent sampling of prompts. Specifically, a directed graph is first constructed to represent unlabeled data. Afterward, the influence of candidate unlabeled subsets is quantified with a diffusion process. A simple yet effective greedy algorithm for unlabeled data selection is lastly introduced. It iteratively selects the data if it provides a maximum marginal gain with respect to quantified influence. Compared with previous efforts on selective annotations, our influence-driven method works in an end-to-end manner, avoids an intractable explicit balance between data diversity and representativeness, and enjoys theoretical support. Experiments confirm the superiority of the proposed method on various benchmarks, achieving better performance under lower time consumption during subset selection.

## 🛠️ **Requirements**

To install requirements:
```setup
conda env create -f ideal.yml
conda activate ideal
cd transformers
pip install -e .
```

It will create the conda environment ideal we used.


## 🚀 How to run?


### **Activate the environment**

```setup
conda activate ideal
```

### **End-to-end pip line for experiments**

a. Perform evaluations on MRPC, SST-5, MNLI, DBpedia, RTE, HellaSwag, and Xsum.
```python
python main.py  --model_cache_dir models 
                --data_cache_dir datasets 
                --task_name mrpc 
                --selective_annotation_method ideal 
                --annotation_size 18
                --cuda_id 0
                --model_name EleutherAI/gpt-j-6B

```

It will run IDEAL on mrpc with GPT-J 6B. The annotation budget is `18`. 


b. Perform evaluations on MWoZ
```python
python main_mowz.py --model_key your_openai_key_here
                    --annotation_size 18
                    --selection_1 ideal
                    --selection_2 similar
                    --cuda_id 0
```

It will run IDEAL on MWoZ dataset with Text-devinci-002. The annotation budget is `18`.

c. Perform evaluations on GeoQuery
```python
python main_geo.py  --model_key your_openai_key_here
                    --annotation_size 18
                    --selective_annotation_method ideal
                    --cuda_id 0
```
It will run IDEAL on GeoQuery dataset with Text-devinci-002. The annotation budget is `18`.


## 📚 License

This code is distributed under an [Apache LICENSE](LICENSE). Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.

## 🌹 Acknowledgement

The code is on the basis of [MetalCL](https://github.com/facebookresearch/MetaICL) and [Vote-k](https://github.com/xlang-ai/icl-selective-annotation). Thanks to all contributors!


## 🤝🏼 Citation
If you find the code is useful in your research, please cite us: 
```
@article{zhang2023ide,
  title={IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models},
  author={Zhang, Shaokun and Xia, Xiaobo and Wang, Zhaoqing and Chen, Ling-Hao and Liu, Jiale and Wu, Qingyun and Liu, Tongliang},
  journal={arXiv preprint arXiv:2310.10873},
  year={2023}
}
```

If you have any question, please contact at: shaokun [DOT] zhang [AT] psu [DOT] edu, xiaoboxia [DOT] uni [AT] gmail [DOT] com.
