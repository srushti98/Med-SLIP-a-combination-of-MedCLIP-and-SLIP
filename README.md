# [MedSLIP: Combining image only self-supervision and image-text self-supervision for MedCLIP](https://drive.google.com/file/d/1GgsMnwQdOBMcaUyRw_vNGr-7DVNVXYaL/view?usp=sharing)

<p align="center"><img src="medslip.png" alt="SLIP framework" width="400"/></p>


## What is this repo about? [[report]](https://drive.google.com/file/d/1GgsMnwQdOBMcaUyRw_vNGr-7DVNVXYaL/view?usp=sharing):
- In the project, I have fine-tuned MedCLIP on a large scale medical image-text pair with SLIP objective.
- Fine-tuning is performed on ROCO dataset which contains 81,825 radiology images, and later it is evaluated on CBIS dataset which contains mammograms of breast cancer for downstream image classification task after linear probing.
- MedSLIP outperforms MedPaLM -M’s benchmark on CBIS by +1% (macro-AUROC) and MedCLIP by +20% (recall - linear probing).

## Results
The following models are finetuned on ROCO and evaluated on CBIS dataset.

### Linear Classification Results on Abnormality Type - Calcification (CBIS dataset)

*Metrics are calculated by macro averaging. ***Official benchmarks- MedPalM-M.*

| Model                          | AUROC* | F1-Score* | Precision* | Recall* | Precision (Malignant) | Recall (Malignant) |
| ------------------------------ | ------ | -------- | ---------- | ------- | --------------------- | ------------------ |
| MedPALM-M (12B)**              | 81.4   | **67.83** | N/A        | N/A     | N/A                   | N/A                |
| MedPALM-M (84B)**              | 82.22  | 63.81    | N/A        | N/A     | N/A                   | N/A                |
| MedPALM-M (562B)**             | 80.9   | 63.03    | N/A        | N/A     | N/A                   | N/A                |
| MedCLIP-ViT                    | 82.22  | 63.52    | **68.37**  | 65.48   | **78.68**             | 37.20              |
| MedCLIP-ResNet                 | 78.18  | 58.88    | 57.91      | 60.84   | 56.6                  | 46.5               |
| MedSLIP-ViT (entire ROCO)      | 80.13  | 64.28    | 66.61      | 65.45   | 71.42                 | 42.63              |
| MedSLIP-ResNet (entire ROCO)   | 75.64  | 57.38    | 56.54      | 59.38   | 59.1                  | 52.7               |
| MedSLIP-ViT (mammo+xray)       | 82.11  | 64.65    | 65.31      | 64.7    | 65.38                 | 52.71              |
| MedSLIP-ResNet (mammo+xray)    | 77.75  | 59.75    | 59.07      | 60.97   | 60.4                  | **62.8**           |
| **MedSLIP-ViT (only mammo)**   | **83.17** | 67.09 | 67.55      | **67.29** | 66.98             | 55.03              |
| MedSLIP-ResNet(only mammo)     | 78.35  | 61.51    | 60.77      | 63.46   | 59.7                  | 62.0               |



### Linear Classification Results on Abnormality Type - MASS (CBIS dataset)

*Metrics are calculated by macro averaging. ***Official benchmarks- MedPalM-M.*

| Model                          | AUROC* | F1-Score* | Precision* | Recall* | Precision (Malignant) | Recall (Malignant) |
| ------------------------------ | ------ | -------- | ---------- | ------- | --------------------- | ------------------ |
| MedPALM-M (12B)**              | 70.11  | 47.23    | N/A        | N/A     | N/A                   | N/A                |
| MedPALM-M (84B)**              | **73.09** | 49.98  | N/A        | N/A     | N/A                   | N/A                |
| MedPALM-M (562B)**             | 73.31  | **51.12** | N/A        | N/A     | N/A                   | N/A                |
| MedCLIP-ViT                    | 64.39  | 40.77    | **54.78**  | 42.64   | 47.9                  | 70.1               |
| MedCLIP-ResNet                 | 57.55  | 35.99    | 35.50      | 38.82   | 43.9                  | 68.0               |
| MedSLIP-ViT (entire ROCO)      | 65.90  | 35.76    | 35.76      | 39.76   | 44.7                  | 69.4               |
| MedSLIP-ResNet (entire ROCO)   | 59.45  | 37.48    | 35.74      | 39.39   | 47.8                  | 51.7               |
| **MedSLIP-ViT (mammo+xray)**   | 67.60  | 42.14    | 44.4       | **44.28** | **52.35**          | **75.51**          |
| MedSLIP-ResNet (mammo+xray)    | 61.42  | 39.95    | 43.49      | 41.35   | 49.2                  | 62.6               |
| MedSLIP-ViT (only mammo)       | 65.61  | 41.19    | 44.08      | 43.15   | 50.96                 | 72.10              |
| MedSLIP-ResNet(only mammo)     | 58.43  | 38.31    | 50.30      | 38.72   | 43.78                 | 55.10              |

## 1. Setup
Install [PyTorch](https://pytorch.org) and [timm](https://github.com/rwightman/pytorch-image-models). 
The code has been tested with CUDA 11.3/CuDNN 8.2.0, PyTorch 1.10.0 and timm 0.5.0.

### 1.1. YFCC15M Setup
Download the [YFCC100M dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).
Our dataloader expects the following dataset directory structure with 100 folders containing 1000 zip archives of 1000 images each.
The concatenation of the folder, archive, and file names is the index of the image (i.e. image 12345678 is stored as `678.jpg` within `12/345.zip`):

```
/path/to/yfcc100m/
├── images/
│   ├── 00/
│   │   └── 000.zip
│   │   │   ├── 000.jpg
│   │   │   │   ...
│   │   │   └── 999.jpg
│   │   ...
│   │   └── 999.zip
│   ...
│   └── 99/
...
```

Prepare the YFCC15M subset metadata pickle:
1. Download and compile a list of downloaded images to `flickr_unique_ids.npy` ([ours](https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy))
2. Download OpenAI's list of captioned YFCC100M images according to instructions [here](https://github.com/openai/CLIP/blob/8cad3a736a833bc4c9b4dd34ef12b52ec0e68856/data/yfcc100m.md)
3. Run `python make_dataset.py` to create the `yfcc15m.pkl` metadata pickle

When pre-training with YFCC15M, set `--dataset yfcc15m --root /path/to/yfcc100m --metadata /path/to/yfcc15m.pkl`.

### 1.2. COCO Captions Setup
Download and unzip the 2017 Train [images](http://images.cocodataset.org/zips/train2017.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
When pre-training on COCO, set `--dataset coco --root /path/to/coco --metadata /path/to/captions_train2017.json`.

### 1.3. Conceptual Captions Setup
[CC3M](https://ai.google.com/research/ConceptualCaptions/download) and [CC12M](https://github.com/google-research-datasets/conceptual-12m) are published as tsv files listing original image urls and processed captions.
Download images and collect the captions of all available images (many will be missing due to broken links) into `cc3m.npy` and `cc12m.npy`.

For CC3M our dataloader expects `cc3m.npy` to contain a NumPy array of dicts in the following format:

```
{
  'image_id': 1510438788,  # local file path relative to root
  'captions': ['large field with pink tulips on a clear sunny summer day with a blue sky']
}
```

For CC12M our dataloader expects `cc12m.npy` to contain a NumPy array of dicts in the following format:

```
{
  'image_name': '0.jpg',  # local file path relative to root
  'image_id': 0,
  'captions': ['Metal Design Within Reach Ivory Slipper Chairs - a Pair For Sale - Image 7 of 10']
}
```

When pre-training on CC3M set `--dataset cc3m --root /path/to/cc3m --metadata /path/to/cc3m.npy`, and whe pre-training on CC12M set `--dataset cc12m --root /path/to/cc12m --metadata /path/to/cc12m.npy`.

### 1.4. RedCaps Setup
[RedCaps](https://redcaps.xyz) is published as a list of JSON annotation files containing image urls and raw/processed captions.
Images can be downloaded from these annotations with a helpful [downloader tool](https://github.com/redcaps-dataset/redcaps-downloader).
Then merge all per-subreddit annotations into a single file with the [combine_captions.py](redcaps/combine_captions.py) script:

```
python redcaps/combine_captions.py --input /path/to/redcaps/annotations --output /path/to/redcaps_v1.json
```

To pre-train on RedCaps set `--dataset redcaps --root /path/to/redcaps --metadata /path/to/redcaps_v1.json`.


### 1.4. Downstream Dataset Setup
Zero-shot (in [main.py](main.py) and [eval_zeroshot.py](eval_zeroshot.py)) and linear (in [main_linear.py](main_linear.py)) evaluations read dataset paths from [dataset_catalog.json](dataset_catalog.json).
Zero-shot evaluations read CLIP's class labels and caption templates from [labels.json](labels.json) and [templates.json](templates.json).
If just pre-training models on YFCC15M, only the ImageNet path is required for model validation between training epochs.
See Section 3 below on zero-shot transfer evaluation for dataset preparation details.

## 2. Pre-training
We use the following pre-training recipes for SLIP, CLIP, and SimCLR.
See [main.py](main.py) for the full list of default arguments.
We use the same lr and wd settings for all model sizes within the same training framework, and different model sizes can be selected by passing in different strings to the `--model` argument such as `SLIP_VITS16` or `SLIP_VITL16`.

In our workflow we use [submitit](https://github.com/facebookincubator/submitit), which interfaces nicely with Slurm.
For local training with the [torchrun](https://pytorch.org/docs/stable/elastic/run.html) utility (supersedes `torch.distributed.launch`), replace `python run_with_submitit.py` with `torchrun --nproc_per_node=8 main.py`. 
Local multi-node training with `torchrun` should also be possible.

We train most of our models on 8x 8-gpu nodes, but training with fewer gpus is possible by reducing the batch size and setting the `--update-freq` argument above 1 to enable gradient accumulation.
Note that gradient accumulation will increase the variance of minibatch statistics and alter the training dynamics of batchnorm, which is used in SLIP and SimCLR.

### SLIP ViT-Base with 8-nodes (batch size 4096)
```
python run_with_submitit.py \
  --root /path/to/yfcc100m \
  --model SLIP_VITB16 \
  --lr 3e-3 --wd 0.1
```

### CLIP ViT-Base with 8-nodes (batch size 4096)
```
python run_with_submitit.py \
  --root /path/to/yfcc100m \
  --model CLIP_VITB16 \
  --lr 5e-4 --wd 0.5
```

### SimCLR ViT-Base with 8-nodes (batch size 4096)
```
python run_with_submitit.py \
  --root /path/to/yfcc100m \
  --model SIMCLR_VITB16 \
  --ssl-mlp-dim 4096 --ssl-emb-dim 256 --ssl-temp 0.1 \
  --lr 3.2e-3 --wd 0.1 
```

Some important arguments:

`--dataset`: pre-training dataset name. choices include `yfcc15m`, `cc12m`, `cc3m`, `coco`.

`--root`: path to dataset root

`--metadata`: path to metadata file (see section 1 for details)

`--ssl-mlp-dim`: hidden dim of SimCLR mlp projection head

`--ssl-emb-dim`: output embed dim of SimCLR mlp projection head

`--ssl-scale`: loss scale for SimCLR objective

`--ssl-temp`: softmax temperature for SimCLR objective

`--batch-size`: number of samples per-device/per-gpu 

`--lr-start`: initial warmup lr

`--lr-end`: minimum final lr

`--update-freq`: optimizer update frequency, i.e. gradient accumulation steps

`--disable-amp`: disable mixed-precision training (requires more memory and compute)

## 3. Evaluation: Zero-shot Transfer
First, prepare additional downstream classification datasets:
- MNIST, CIFAR-10/100, STL-10: Automatic download via [torchvision datasets](https://pytorch.org/vision/stable/datasets.html)
- HatefulMemes: Manual download from [official website](https://hatefulmemeschallenge.com/#download) and sort images according to `train.jsonl`/`dev.jsonl` into train/dev folder
- Rendered SST2, Country211: Manual download from [CLIP repo](https://github.com/openai/CLIP/tree/main/data)
- Other datasets: Use scripts from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets)

Then set all dataset paths in [dataset_catalog.json](dataset_catalog.json).

Evaluate zero-shot transfer to various classification benchmarks with [eval_zeroshot.py](eval_zeroshot.py), which reads labels and templates from [labels.json](labels.json)/[templates.json](templates.json) and dataset paths from [dataset_catalog.json](dataset_catalog.json). Inference is performed with a single gpu. By default, the script iterates through all datasets in [dataset_catalog.json](dataset_catalog.json) and evaluates zero-shot in order. Evaluation can be limited to a subset of datasets by replacing `for d in datasets:` with `for d in ['imagenet']:` on line 78.

```
python eval_zeroshot.py --resume /path/to/checkpoint.pt
```

## 4. Evaluation: Linear Classification
We use a modified version of the MoCo v3 ImageNet linear classification script, [main_linear.py](main_linear.py).
We use the same single node 8-gpu recipe for all model sizes.
See [main_linear.py](main_linear.py) for the full list of default arguments.
As with pre-training, our workflow uses [submitit](https://github.com/facebookincubator/submitit).
For local training with [torchrun](https://pytorch.org/docs/stable/elastic/run.html), replace `python run_with_submitit_linear.py` with `torchrun --nproc_per_node=8 main_linear.py`. 
This script reads the ImageNet dataset path from the dataset catalog ([dataset_catalog.json](dataset_catalog.json)), which must be set properly before training.

```
python run_with_submitit_linear.py  \
  --arch vit_base_patch16_224 --dataset imagenet \
  --pretrained /path/to/checkpoint.pt
```

To evaluate linear classification on other datasets, set `--dataset` to the corresponding dataset name listed in [dataset_catalog.json](dataset_catalog.json).

## 5. Evaluation: End-to-End Finetuning
We use a modified version of the ImageNet finetuning script from [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit).
Our code has been tested with commit `f8f3df8`.
We have removed the explicit torch, torchvision, and timm dependencies from [beit_finetuning/requirements.txt](beit_finetuning/requirements.txt), as they conflict with the versions used in our SLIP code (CUDA 11.3/CuDNN 8.2.0, PyTorch 1.10.0 and timm 0.5.0).
The fintuning code has been modified and tested to work with these versions.

### 5.1. Setup
To evaluate end-to-end finetuning on ImageNet, first clone the BeiT repo and checkout the correct commit:

```
git clone git@github.com:microsoft/unilm.git
cd unilm/beit
git checkout f8f3df8
```

Now copy over modified files from our [beit_finetuning](beit_finetuning) directory:

```
cp beit_finetuning/* unilm/beit
cd unilm/beit
```

Install pip dependencies and Nvidia Apex:

```
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


### 5.2. Commands
As with pre-training, our workflow uses [submitit](https://github.com/facebookincubator/submitit).
For local training with [torchrun](https://pytorch.org/docs/stable/elastic/run.html), replace `python run_with_submitit_finetune.py` with `torchrun --nproc_per_node=8 run_class_finetuning.py`. 
We established finetuning recipes based on the BeiT recipes with some light additional hyperparameter tuning.
We increase regularization with model size: ViT-S uses drop_path=0 and layer_decay=0.65, ViT-B uses drop_path=0.1 and layer_decay=0.65, and ViT-L uses drop_path=0.1 and layer_decay=0.75.
Note the use of the `--finetune` argument instead of `--resume`.

### ViT-Small (MoCo v3 version w/ 12 vs. 6 heads)

```
python run_with_submitit_finetune.py \
    --batch_size 128 --enable_deepspeed \
    --epochs 100 --warmup_epochs 20 \
    --model beit_small_patch16_224 --nb_classes 1000 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --mixup 0.8 --cutmix 1 \
    --layer_scale_init_value 0 \
    --lr 4e-3 --drop_path 0 --layer_decay 0.65 \
    --output_dir /path/to/output_dir --finetune /path/to/checkpoint.pt
```

### ViT-Base

```
python run_with_submitit_finetune.py \
    --batch_size 128 --enable_deepspeed \
    --epochs 100 --warmup_epochs 20 \
    --model beit_base_patch16_224 --nb_classes 1000 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --mixup 0.8 --cutmix 1 \
    --layer_scale_init_value 0 \
    --lr 4e-3 --drop_path 0.1 --layer_decay 0.65 \
    --output_dir /path/to/output_dir --finetune /path/to/checkpoint.pt
```

### ViT-Large

```
python run_with_submitit_finetune.py \
    --batch_size 128 --enable_deepspeed \
    --epochs 50 --warmup_epochs 5 \
    --model beit_large_patch16_224 --nb_classes 1000 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --mixup 0.8 --cutmix 1 \
    --layer_scale_init_value 0 \
    --lr 4e-3 --drop_path 0.1 --layer_decay 0.75 \
    --output_dir /path/to/output_dir --finetune /path/to/checkpoint.pt
```


### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.
