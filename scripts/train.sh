python src/data/load_dataset.py -cf params.yaml
python yolov7/train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 1 --data data/data.yaml --weights models/yolov7.pt --device cpu --project models
