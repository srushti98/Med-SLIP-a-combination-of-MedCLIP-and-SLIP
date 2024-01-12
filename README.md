# [MedSLIP: Combining image only self-supervision and image-text self-supervision for MedCLIP](https://drive.google.com/file/d/1GgsMnwQdOBMcaUyRw_vNGr-7DVNVXYaL/view?usp=sharing)

<p align="center"><img src="medslip.png" alt="SLIP framework" width="400"/></p>


## What is this repo about? [[report]](https://drive.google.com/file/d/1GgsMnwQdOBMcaUyRw_vNGr-7DVNVXYaL/view?usp=sharing):
- In the project, I have fine-tuned MedCLIP on a large scale medical image-text pair with SLIP objective.
- Fine-tuning is performed on [ROCO dataset](https://github.com/razorx89/roco-dataset) which contains 81,825 radiology images, and later it is evaluated on [CBIS dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) which contains mammograms of breast cancer for downstream image classification task after linear probing.
- MedSLIP outperforms [MedPaLM-M](https://arxiv.org/abs/2307.14334)’s benchmark on CBIS by +1% (macro-AUROC) and [MedCLIP](https://github.com/RyanWangZf/MedCLIP/tree/main/medclip) by +20% (recall - linear probing).

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

## 2. Fine-Tuning

We use the SLIP objective.
See [main.py](main.py) for the full list of default arguments.


Some important arguments:

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


## 4. Evaluation: Linear Classification

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



### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.
