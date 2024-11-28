CUDA_VISIBLE_DEVICES=2 python apply_net.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0

cd evaluator/

CUDA_VISIBLE_DEVICES=2 python eval.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

CUDA_VISIBLE_DEVICES=2 python aose.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

CUDA_VISIBLE_DEVICES=2 python WI.py --dataset-dir /data/datasets/Detection/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

# cd ..   