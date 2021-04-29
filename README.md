## Object Detection

The code in `object_detction.ipynb` uses the pretrained [FasterRCNN](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html) model in PyTorch to detect objects belonging to [80 classes](https://cocodataset.org/#explore) as defined in COCO (2014) dataset in `coco_labels.txt`. As per our need, these 80 classes are further grouped into 10 super classes using a python dictionary and written in `coco_super_clasess.json`. The code finally displays these super classes in an image. The detected bounding boxes can be displayed by turning on the 'display_image' flag.


Super-class         | COCO Class Label
------------------- | -------------
person              | person
modes of transport  | bicycle, car, motorcycle, airplane, bus, train, truck, boat
street view         | traffic light, fire hydrant, stop sign, parking meter, bench
animals             | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
sports              | frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
food                | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
kitchen             | wine glass, cup, fork, knife, spoon, bowl
indoor              | couch, potted plant, bed, dining table, toilet, sink, clock, vase
electronis          | tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, refrigerator, hair drier
misc                | book, scissors, teddy bear, toothbrush, tie, backpack, umbrella, handbag, suitcase, chair, bottle

## Hyperparameter tuning

The code detects objects with probability score > 0.6. This method of filtering can be imropved using non-maximum supression [nms.py and iou.py](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/metrics). 
The ideal threshold values can be found by hyperparameter tuning on the COCO validation set and matching the mean average precision (map.py) to SOTA values.

## Training with custom dataset

The code in `object_detection_custom_dataset.ipnyb` loads the [Flicker logos 27k](http://image.ntua.gr/iva/datasets/flickr_logos/) dataset in PyTorch and fine-tunes the fasterRCNN model on this dataset. (For this part of the code any batch size above 8 gives a CUDA out of memory error on Google Colab. It needs to be further trained and tested using more powerful GPUs.)
