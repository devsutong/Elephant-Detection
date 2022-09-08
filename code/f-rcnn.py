import torch, torchvision

print(torch.__version__, torch.cuda.is_available())



import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
  
# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import matplotlib.pyplot as plt

# im = cv2.imread("../img/test_img/white.png")
im = cv2.imread("../img/test_img/1.jpeg")


cfg = get_cfg()
cfg.merge_from_file(("../config_files/k-config.yaml"))
cfg.MODEL.DEVICE = "cpu"
# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(im)
print("------------Result----------------")
print(outputs)
print(outputs["instances"].pred_classes)
c = outputs["instances"].pred_classes
x = c.numpy()
print(x)
for a in x:
    print(x[a])
print("------------Result----------------")

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)