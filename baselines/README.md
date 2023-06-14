# Baselines for LSME

## Pretrained Weights
DINOv2 ViT B/14 - ABC
```
https://www.dropbox.com/s/fhw2j37wxszicpc/DINOv2B14_ABC.pt?dl=0
```
DINOv2 ViT S/14 - ABC
```
https://www.dropbox.com/s/clysymckhuzw7n2/DINOv2S14_ABC.pt?dl=0
```
DINOv1 ViT S/8 - ABC
```
https://www.dropbox.com/s/vkv6c6koyyom70j/DINOv1S8_ABC.pt?dl=0
```

## Pretrained Data
Renderings for ABC can be downloaded from
```
https://www.dropbox.com/sh/rlldg6ldr7073i2/AAAjV3UhOaA7fDpvcxRPmho4a?dl=0
```

## LSME Data
Renderings for multi-object variants on Toys4K
```
https://www.dropbox.com/s/sil2wfaawwgetmv/toys_Multi_Object_data.tar.xz?dl=0
```
Data for Categ-SObject
```
https://www.dropbox.com/s/wlbcmq9ohb4euwn/toys_query_Categ_SObject.xz?dl=0
https://www.dropbox.com/s/qmcjcttyhcjkfta/toys_support_Categ_SObject.xz?dl=0
```
Data for Categ-SObject-PoseVar
```
https://www.dropbox.com/s/es15lfdia4l1pbe/toys_query_Categ_SObject_PoseVar.tar.xz?dl=0
https://www.dropbox.com/s/rlhme8uwq3kjqft/toys_support_Categ_SObject_PoseVar.tar.xz?dl=0
```
Predicted instance masks
```
https://www.dropbox.com/s/2o3fwxlw1k37ewy/support.tar.xz?dl=0
https://www.dropbox.com/s/zm8y6wz0kqzb69f/query.tar.xz?dl=0
```

## Installation
```
conda env create -f environment.yml
conda activate lsme

pip install -e .
```

## Low-shot Evaluation on LSME setting
Download pretrained DINOv2 ViT B/14 - ABC weights from [here](https://www.dropbox.com/s/fhw2j37wxszicpc/DINOv2B14_ABC.pt?dl=0)

Download renderings from [here](https://www.dropbox.com/s/sil2wfaawwgetmv/toys_Multi_Object_data.tar.xz?dl=0) and predicted instance masks for [support scenes](https://www.dropbox.com/s/2o3fwxlw1k37ewy/support.tar.xz?dl=0) and [query scenes](https://www.dropbox.com/s/zm8y6wz0kqzb69f/query.tar.xz?dl=0)

Adjust the data paths in `paths.py`

Run the following command
```
python scripts/test.py --cfg configs/contrastive_global/vispe_moco_ABC.yaml --ckpt /path/to/pretrained/model
```



