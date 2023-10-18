# IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models

**Official implementation for paper [IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models](https://arxiv.org/abs/2310.10873).**

## Introduction

In-context learning is a promising paradigm that utilizes In-context examples as prompts for the predictions of large language models. These prompts are crucial for achieving strong performance. However, since the prompts need to be sampled from a large volume of annotated examples, finding the right prompt may result in high annotation costs. To address this challenge, this paper introduces an influence-driven selective annotation method that aims to minimize annotation costs while improving the quality of In-context examples. The essence of our method is
to select a pivotal subset from a large-scale unlabeled data pool to annotate for the subsequent sampling of prompts. Specifically, a directed graph is first constructed to represent unlabeled data. Afterward, the influence of candidate unlabeled subsets is quantified with a diffusion process. A simple yet effective greedy algorithm for unlabeled data selection is lastly introduced. It iteratively selects the data if it provides a maximum marginal gain with respect to quantified influence. Compared with previous efforts on selective annotations, our influence-driven method works in an end-to-end manner, avoids an intractable explicit balance between data diversity and representativeness, and enjoys theoretical support. Experiments confirm the superiority of the proposed method on various benchmarks, achieving better performance under lower time consumption during subset selection.

## Experiments

### **Requirements**

To install requirements:
```setup
conda env create -f ideal.yml
conda activate ideal
cd transformers
pip install -e .
```

It will create the conda environment ideal we used.

### **How to run** 

1. **Activate the environment**

    ```setup
    conda activate ideal
    ```
1. **End-to-end pip line for experiments**

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

    It will run IDEAL on mrpc with GPT-J 6B. The annotation budget is 18. 


    b. Perform evaluations on MWoZ
    ```python
	python main_mowz.py --model_key your_openai_key_here
                        --annotation_size 18
                        --selection_1 ideal
                        --selection_2 similar
                        --cuda_id 0
    ```

    It will run IDEAL on MWoZ dataset with Text-devinci-002. The annotation budget is 18.

    c. Perform evaluations on GeoQuery
    ```python
	python main_geo.py  --model_key your_openai_key_here
                        --annotation_size 18
                        --selective_annotation_method ideal
                        --cuda_id 0
    ```
    It will run IDEAL on GeoQuery dataset with Text-devinci-002. The annotation budget is 18.


## References
[1]: Min, Sewon, et al. "Metaicl: Learning to learn in context." arXiv preprint arXiv:2110.15943 (2021).

[2]: Su, Hongjin, et al. "Selective annotation makes language models better few-shot learners." arXiv preprint arXiv:2209.01975 (2022).


## Citation
If you find the code is useful in your research, please cite us
```
@article{zhang2023ide,
  title={IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models},
  author={Zhang, Shaokun and Xia, Xiaobo and Wang, Zhaoqing and Chen, Ling-Hao and Liu, Jiale and Wu, Qingyun and Liu, Tongliang},
  journal={arXiv preprint arXiv:2310.10873},
  year={2023}
}
```

