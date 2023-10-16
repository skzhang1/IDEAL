# IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models

## Introduction

Official implementation for paper **IDEAL: Influence-Driven Selective Annotations Empower In-context Learners in Large Language Models**. 
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
[1]: Min, S., Lewis, M., Zettlemoyer, L., & Hajishirzi, H. (2021). Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943.

[2]: Su, H., Kasai, J., Wu, C. H., Shi, W., Wang, T., Xin, J., ... & Yu, T. (2022). Selective annotation makes language models better few-shot learners. arXiv preprint arXiv:2209.01975.


