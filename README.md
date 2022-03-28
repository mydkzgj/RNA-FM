# RNA-FM
This repository contains codes and pre-trained models for **RNA foundation model (RNA-FM)**.
**RNA-FM outperforms all tested single-sequence RNA language models across a variety of structure prediction tasks as well as several function-related tasks.**
You can find more details about **RNA-FM** in our paper, ["" (xx et al., 2022).](https://www.runoob.com)

<details><summary>Citation</summary>

```bibtex
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={bioRxiv}
}
```
</details>

<details><summary>Table of contents</summary>
  
- [Comparison to related works](#perf-related)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Compute embeddings in bulk from FASTA](#bulk-fasta)
  - [Notebooks](#notebooks)
- [Benchmarks](#perf)
  - [Comparison on several tasks](#perf-related)
- [Available Models and Datasets](#available)
  - [Pre-trained Models](#available-models)
  - [ESM Structural Split Dataset](#available-esmssd)
  - [Pre-training Dataset Split](#available-pretraining-split)
- [Citations](#citations)
- [License](#license)
</details>

<details><summary>What's New</summary>
  
- Mar 2022: RNA-FM added (see [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1)).
  
</details>

## Create Environment with Conda
```
conda env create -f environment.yml
```
Then activate the "RNA-FM" environment
```
conda activate RNA-FM
cd ./redevelop
```

## Access pre-trained models.
download from [this gdrive link]() and place them into the `data` folder.

## Apply RNA-FM.
### 1. Embedding Extraction
```
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1 --save_embeddings
```
--config="pretrained/extract_embedding.yml" --data_path="./data/examples/example.fasta" --save_dir="./resuts" --save_frequency 1 --save_embeddings

### 2. Downstream Prediction - RNA secondary structure
```
python launch/predict.py --config="pretrained/ss_prediction.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1
```
--config="pretrained/ss_prediction.yml" --data_path="./data/examples/example.fasta" --save_dir="./resuts" --save_frequency 1
