﻿﻿﻿# Adaptive Unknown Object Detector: Revisiting Objectness


#### [Haomiao Liu]<sup>\*</sup>, [Hao Xu]<sup>\*</sup>, [Chuhuai Yue], [Bo Ma] ####

(* denotes equal contribution)

# Abstract

Unknown object detection requires using a training set with only known objects labeled to build a model that can be generalized to unknown objects. A key element in achieving this is effective objectness learning from known objects to represent unknown ones. However, inflexible and non-adaptive objectness learning in existing methods compromises unknown object detection. 
In this paper, we propose the Adaptive Unknown Object Detector (AUD) to address the above issues. First, we propose the Multi-scale Feature Adaptive Fusion Module (MFAFM), which adaptively fuses cross-scale features based on attentional weights to provide rich and discriminative semantic information for objectness learning, enhancing the recall of unknown objects with semantic and scale variations. Second, we introduce the Adaptive Objectness Score (AOS), which flexibly learns generalized objectness knowledge from the positional relations of known objects to accurately localize unknown object boundaries. In addition, we design a Boxes Adaptive Determination (BAD) strategy as post-processing to judge and retain the prediction results.Finally, the experimental results demonstrate that our method significantly outperforms existing state-of-the-art methods, achieving 18.3\% and 11.4\% absolute gains in unknown precision rate on the COCO-OOD and COCO-Mixed benchmarks, respectively.



</div>

# Requirements
```bash
pip install -r requirements.txt
```

In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Dataset Preparation

The datasets can be downloaded using this [link](https://drive.google.com/drive/folders/1Mh4xseUq8jJP129uqCvG9cSLdjqdl0Jo?usp=sharing).

**PASCAL VOC**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

Please download the JPEGImages data from the [link](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing) provided by [VOS](https://github.com/deeplearning-wisc/vos).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         ├── voc0712_train_completely_annotation200.json
         └── val_coco_format.json

**COCO**

Please put the corresponding json files in Google Cloud Disk into ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_coco_ood.json
            ├── instances_val2017_mixed_ID.json
            └── instances_val2017_mixed_OOD.json
         ├── train2017
         └── val2017

# Training
```bash
python train_net.py --dataset-dir VOC_DATASET_ROOT --num-gpus 2 --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --random-seed 0 --resume
```
The pretrained models for Pascal-VOC can be downloaded from [Here]( https://pan.baidu.com/s/1LYnIdAx9ZYCmGP3ZyQYDDQ?pwd=rqdw). Please put the model in ./detection/data/VOC-Detection/faster-rcnn/IOU_FFN/random_seed_0/.

# Pretesting
The function of this process is to obtain the threshold, which only uses part of the training data.
```bash
sh pretest.sh
```

# Evaluation on the VOC
```bash
python apply_net.py --dataset-dir VOC_DATASET_ROOT --test-dataset voc_custom_val  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0
```

# Evaluation on the COCO-OOD
```bash
sh test_ood.sh
```

# Evaluation on the COCO-Mix

```bash
sh test_mixed.sh
```

# Visualize prediction results
```bash
sh vis.sh
```

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

