# HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

Official implementation of "HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks" (ICLR 2025).

## Abstract

Mechanistic interpretability has made great strides in identifying neural network features (e.g., directions in hidden activation space) that mediate concepts (e.g., the birth year of a person) and enable predictable manipulation. Distributed alignment search (DAS) leverages supervision from counterfactual data to learn concept features within hidden states, but DAS assumes we can afford to conduct a brute force search over potential feature locations. To address this, we present HyperDAS, a transformer-based hypernetwork architecture that (1) automatically locates the token-positions of the residual stream that a concept is realized in and (2) constructs features of those residual stream vectors for the concept. In experiments with Llama3-8B, HyperDAS achieves state-of-the-art performance on the RAVEL benchmark for disentangling concepts in hidden states.

## Overview

<p align="center">
  <img src="assets/hyperdas_framework.png" width="800px">
</p>

The HyperDAS framework, used here to find the features that mediate the concept of "country":

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
conda create -n hyperdas python=3.9
conda activate hyperdas

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from hyperdas import HyperDAS
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLama model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

# Initialize HyperDAS with target layer
hyperdas = HyperDAS(
    model=model,
    layer=15,  # Best performance is usually at middle layers
    feature_dim=128
)

# Train on RAVEL dataset
hyperdas.train(
    dataset_path="data/ravel/city",
    epochs=5,
    batch_size=32
)

# Perform intervention
base_prompt = "Vienna is in"
counterfactual_prompt = "I love Paris"
concept = "the country of a city"

output = hyperdas.intervene(
    base_prompt=base_prompt,
    counterfactual_prompt=counterfactual_prompt,
    concept=concept
)

print(f"Original output: {model.generate(base_prompt)}")
print(f"Intervention output: {output}")
```

## Dataset

We evaluate HyperDAS on the RAVEL benchmark, which contains five domains:
- Cities
- Nobel laureates
- Occupations
- Physical objects
- Verbs

Each domain has multiple attributes that can be disentangled (e.g., for cities: country, language, continent, timezone, latitude, longitude).

You can download the RAVEL dataset using:

```bash
python scripts/download_ravel.py
```

## Experiments

To reproduce our experiments on RAVEL:

```bash
# Train HyperDAS on a specific domain (e.g., city)
python train.py --domain city --layer 15 --feature_dim 128 --epochs 5

# Train a single HyperDAS model on all domains
python train.py --domain all --layer 15 --feature_dim 128 --epochs 5

# Evaluate HyperDAS
python evaluate.py --model_path checkpoints/hyperdas_city_l15.pt --domain city
```

## Results

HyperDAS achieves state-of-the-art performance on RAVEL:

| Method | City | Nobel Laureate | Occupation | Physical Object | Verb | Average |
|--------|------|----------------|------------|-----------------|------|---------|
| MDAS | 66.9 | 74.8 | 69.4 | 91.5 | 77.0 | 76.0 |
| HyperDAS | 82.4 | 75.3 | 74.8 | 95.0 | 96.0 | 84.7 |

## Citation

```bibtex
@inproceedings{sun2025hyperdas,
  title={HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks},
  author={Sun, Jiuding and Huang, Jing and Baskaran, Sidharth and D'Oosterlinck, Karel and Potts, Christopher and Sklar, Michael and Geiger, Atticus},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License

MIT

## Acknowledgements

This research was in part supported by a grant from Open Philanthropy.
