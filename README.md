# HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

Official implementation of "[HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks](https://arxiv.org/pdf/2503.10894)" (ICLR 2025).


## Abstract

Mechanistic interpretability has made great strides in identifying neural network features (e.g., directions in hidden activation space) that mediate concepts (e.g., the birth year of a person) and enable predictable manipulation. Distributed alignment search (DAS) leverages supervision from counterfactual data to learn concept features within hidden states, but DAS assumes we can afford to conduct a brute force search over potential feature locations. To address this, we present HyperDAS, a transformer-based hypernetwork architecture that (1) automatically locates the token-positions of the residual stream that a concept is realized in and (2) constructs features of those residual stream vectors for the concept. In experiments with Llama3-8B, HyperDAS achieves state-of-the-art performance on the RAVEL benchmark for disentangling concepts in hidden states.

## Overview

<p align="center">
  <img src="main_fig.png" width="800px">
</p>

The HyperDAS is a highly modulized framework, used here to search for the features in LLMs that mediate the concept of "country" automatically:

1. **Concept Encoding**: A natural language description that specifies which concept to localize, "The country of a city", is encoded by a transformer hypernetwork with two additional cross-attention blocks attending to the hidden states of the target LM prompted with the base text "Vienna is in" and the counterfactual text "I love Paris".

2. **Selecting Token-Positions**: With the encoding from step 1 as a query, HyperDAS selects the tokens "nna" and "Paris" as the location of the concept "country" for the base and counterfactual, respectively.

3. **Identifying a Subspace**: With the representation from step 1 as the encoding, HyperDAS constructs a matrix whose orthogonal columns are the features for "country".

4. **Interchange Intervention**: With the token-positions from step 2 and subspace from step 3, HyperDAS performs an intervention by patching the subspace of the hidden vector for the token "nna" to the value it takes on in the hidden vector for the token "Paris", leading the model to predict "France" from the base prompt "Vienna is in".

## Installation

```bash
# Clone the repository
git clone https://github.com/jiudingsun01/HyperDAS.git
cd HyperDAS

# Create a conda environment
conda env create -f environment.yml
conda activate hypernet
```

## Quick Start

Checkout the file [demo.ipynb](./demo.ipynb) for a quick tutorial on how to generate dataset and how to train your own interpreter model over counterfactual data.

```python
from src.hyperdas.llama3.modules import {
    LlamaInterpreterConfig, 
    LlamaInterpreter
}

config = LlamaInterpreterConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
config.name_or_path = "meta-llama/Meta-Llama-3-8B"
config.torch_dtype = torch.bfloat16

# the number of attention heads
config.num_editing_heads = 32
# the number of layer 
config.chop_editor_at_layer = 4
# the layer of the target model to apply the intervention
config.intervention_layer = 20 
config._attn_implementation = 'eager'
config.initialize_from_scratch = True 

interpreter = LlamaInterpreter(
    config, 
    subspace_module="ReflectSelect",
    das_dimension=2,
)

interpreter = interpreter.to("cuda")
```

## Dataset

We evaluate HyperDAS on the RAVEL benchmark, which contains five domains:
- Cities
- Nobel laureates
- Occupations
- Physical objects
- Verbs

Each domain has multiple attributes that can be disentangled (e.g., for cities: country, language, continent, timezone, latitude, longitude).

For more informatio on the benchmark, please checkout their [GitHub](https://github.com/explanare/ravel) or [paper](https://arxiv.org/pdf/2402.17700)



## Experiments

To reproduce our experiments on RAVEL, simply do:

```bash
# Train HyperDAS on RAVEL
python train.py --intervention_layer 15 --das_dimension 128 --save_model --save_dir "/Path/To/Your/Dir"

# Evaluate HyperDAS on RAVEL
python inference.py --checkpoint_path "/Path/To/Your/Dir"
```

## Citation
Here is the citation to our paper:
```bibtex
@inproceedings{sun2025hyperdas,
  title={HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks},
  author={Sun, Jiuding and Huang, Jing and Baskaran, Sidharth and D'Oosterlinck, Karel and Potts, Christopher and Sklar, Michael and Geiger, Atticus},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

If you use the `RAVEL` benchmark, please also consider citing the following:

```
@inproceedings{huang-etal-2024-ravel,
    title = "{RAVEL}: Evaluating Interpretability Methods on Disentangling Language Model Representations",
    author = "Huang, Jing  and
      Wu, Zhengxuan  and
      Potts, Christopher  and
      Geva, Mor  and
      Geiger, Atticus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.470",
    pages = "8669--8687",
}

```

If you use the `pyvene` framework, please also consider citing the following:

```
@inproceedings{wu-etal-2024-pyvene,
    title = "pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions",
    author = "Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah and Manning, Christopher and Potts, Christopher",
    editor = "Chang, Kai-Wei and Lee, Annie and Rajani, Nazneen",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-demo.16",
    pages = "158--165",
}
```