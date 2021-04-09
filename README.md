# ObjectDetection
The code in **object_detction.ipynb** can be used to detect objects belonging to 80 different classes as defined in COCO (2014) dataset (_coco_labels.txt_). As per our need, these 80 classes are further grouped into 10 super classes using a python dictionary and written in _coco_super_clasess.json_. The code finally displays these super classes in an image. The detected bounding boxes can be displayed by turning on the 'display_image' flag.
The code detects objects with probability score > 0.6. This method of filtering can be imropved using non-maximum supression (nms.py and iou.py). 
The ideal threshold values for this can be found by hyperparameter tuning on the COCO validation set and matching the mean average precision (map.py) to SOTA values.

The code in **object_detection_custom_dataset.ipnyb** loads the "flicker logos 27k" dataset and fine-tunes the fasterRCNN model on this dataset. 
