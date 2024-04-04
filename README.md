# Analyzing LLMs for Math Word Problems
Paper: [What Makes Math Word Problems Challenging for LLMs?](https://arxiv.org/abs/2403.11369) [NAACL 2024]

## Overview

We formulate and investigate two research questions: 

 1. Which characteristics of an input math word question (sampled from [GSM8K](https://arxiv.org/abs/2110.14168)) make it complex for an LLM?
 2. Based on these characteristics, can we predict whether a particular LLM will be able to solve specific input MWPs correctly?

This repository contains the raw data, the feature data extracted, and code to reproduce and extend the subsequent analysis.

## Usage

### Installation

```bash
git clone https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps.git
cd analyzing-llms-for-mwps
pip install -r requirements.txt
```

### Collect LLM Responses

1. Build query prompts from GSM8K questions by running the notebook: [`./code/llm_querying/build_query_prompts.ipynb`](https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/blob/main/code/llm_querying/build_query_prompts.ipynb)
2. Run [`./code/llm_querying/vllm_query.py`](https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/blob/main/code/llm_querying/vllm_query.py) for querying all prompts for one LLM at a time:
   ```bash
	python3 std_query_vllm.py \
	--model-name "meta-llama/Llama-2-13b-chat-hf" \
	--batch-size 1 \
	--query-limit -1 \
	--n-seq 1 \
	--seed $RANDOM \
	--temperature 0.8 \
	--repetition-penalty 1.0 \
	--max-len 2000 \
	--query-paths ../../data/query_datasets/gsm8k_queries.json
   ```

[OR]

### Import Response Data
The LLM response data used for our work is available [here](https://drive.google.com/file/d/1A2N2hrVjuKc2mj2Lf_ew3BmpZ5rBoGRu/view?usp=sharing).
Download the zip, extract, and replace the contents into the `./data` folder for further processing.

### Extracting Features, Training & Evaluating Classifiers
1. (Optional) Specify HuggingFace Access Token at [here](https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/blob/cac904cb4b84293ada10283a650608c02d6e7c88/code/classifier_based_analysis/utils.py#L4) for accessing restricted models like LLama2.
2. Run the notebook [`./code/classifier_based_analysis/predicting_success_rate.ipynb`](https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/blob/main/code/classifier_based_analysis/predicting_success_rate.ipynb) to:
	1. Extract linguistic, math, and world knowledge features from LLM responses on GSM8K.
	2. Generate relevant feature distribution statistics.
	3. Train statistical classifiers on extracted features to predict. which questions are always or never solved correctly by LLMs.

### Supported LLMs

| LLM Model | HuggingFace Model Name | Pass@1 | Success Rate
|--|--|--|--|
| Llama2-13B | `meta-llama/Llama-2-13b-chat-hf` | 28.70 | 37.24
| Llama2-70B | `meta-llama/Llama-2-70b-chat-hf` | 56.80 | 56.09
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.2` | 40.03 | 36.27
| MetaMath-13B | `meta-math/MetaMath-13B-V1.0` | 72.30 | 63.73


### Classifier Models
1. Logistic Regression
2. Decision Tree
3. Random Forest 
<img width="800" alt="image" src="https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/assets/47175964/d34f773a-2d9d-46a7-b063-a1f1a00672d9">

### Feature Set

Our work proposes a total of 23 features spanning three types: Linguistic (L), Math (M), and World Knowledge (W).
A detailed description of each feature and corresponding Python functions for extraction is covered in [`./code/classifier_based_analysis/feature_extraction.py`](https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/blob/main/code/classifier_based_analysis/feature_extraction.py).

<img width="700" alt="image" src="https://github.com/kvadityasrivatsa/analyzing-llms-for-mwps/assets/47175964/62e97b15-54aa-43cc-8609-c85d700e5565">

## Citation

```bibtex
@misc{srivatsa2024makes,
      title={What Makes Math Word Problems Challenging for LLMs?}, 
      author={KV Aditya Srivatsa and Ekaterina Kochmar},
      year={2024},
      eprint={2403.11369},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```






