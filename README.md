# RNA-FM
This repository contains codes and pre-trained models for **RNA foundation model (RNA-FM)**.
**RNA-FM outperforms all tested single-sequence RNA language models across a variety of structure prediction tasks as well as several function-related tasks.**
You can find more details about **RNA-FM** in our paper, ["" (xx et al., 2022).](https://www.runoob.com)

<details><summary>Citation</summary>

```bibtex
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}
```
</details>

<details><summary>Table of contents</summary>
  
- [Setup Environment](#Setup_Environment)
- [Pre-trained Models](#Available_Pretrained_Models)
- [Usage](#usage)
  - [RNA-FM Embedding Generation](#RNA-FM_Embedding_Generation)
  - [RNA Secondary Structure Prediction](#RNA_Secondary_Structure_Prediction)
- [Citations](#citations)
- [License](#license)
</details>

<details><summary>What's New</summary>
  
- Mar 2022: RNA-FM added (see [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1)).
  
</details>

## Create Environment with Conda <a name="Setup_Environment"></a>
```
conda env create -f environment.yml
```
Then activate the "RNA-FM" environment
```
conda activate RNA-FM
cd ./redevelop
```

## Access pre-trained models. <a name="Available_Pretrained_Models"></a>
download from [this gdrive link](https://drive.google.com/drive/folders/1fWePKPQPFlQNEyJEgmJiGLurDYFD6KDI?usp=sharing) and place them into the `data` folder.

## Apply RNA-FM. <a name="Usage"></a>
### 1. Embedding Extraction. <a name="RNA-FM_Embedding_Generation"></a>
```
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1 --save_embeddings
```

### 2. Downstream Prediction - RNA secondary structure. <a name="RNA_Secondary_Structure_Prediction"></a>
```
python launch/predict.py --config="pretrained/ss_prediction.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1
```
## Citations <a name="citations"></a>

If you find the models useful in your research, we ask that you cite the relevant paper:

For RNA-FM:

```bibtex
@article{meier2021language,
  author = {Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alexander},
  title = {Language models enable zero-shot prediction of the effects of mutations on protein function},
  year={2021},
  doi={10.1101/2021.07.09.450648},
  url={https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1},
  journal={bioRxiv}
}
```

The model of this code builds on the [esm](https://github.com/facebookresearch/esm) sequence modeling framework. 
And we use [fairseq](https://github.com/pytorch/fairseq) sequence modeling framework to train our RNA language modeling.
We very appreciate these two excellent works!

## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.
