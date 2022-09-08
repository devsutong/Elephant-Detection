import time
import sys
import cv2
import numpy as np
import winsound
import threading


"""FRCNN"""
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


cfg = get_cfg()
cfg.merge_from_file(("../config_files/k-config.yaml"))
cfg.MODEL.DEVICE = "cpu"
# Create predictor
predictor = DefaultPredictor(cfg)
FRCNN_ACTIVITY = False
"""FRCNN"""

frame_count = 0
total_frames = 0

def predict_frcnn(im):
    FRCNN_ACTIVITY = True
    outputs = predictor(im)
    classes_result = (outputs["instances"].pred_classes).numpy()
    for class_result in classes_result:
        if class_result == 0:
            winsound.Beep(250, 1000)
    FRCNN_ACTIVITY = False


def build_model(is_cuda):
    net = cv2.dnn.readNet(model_path)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture():
    # capture = cv2.VideoCapture(0) 
    # capture = cv2.VideoCapture("Y:/8th Semester/Major Project/video/Elephants through webcam TallinnZoo2021.mp4")
    # capture = cv2.VideoCapture("Y:/8th Semester/Major Project/video/240pxelephant.mp4")
    capture = cv2.VideoCapture("../video/zoo.mp4")


    return capture

def load_classes():
    class_list = []
    with open("../config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
            

    return result_class_ids, result_confidences, result_boxes

def generate_frame():
    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1
    while True:

        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = unwrap_detection(inputImage, outs[0])

        frame_count += 1
        total_frames += 1

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
            if classid == 0:

                """Start F-RCNN"""

                # import detectron2
                # from detectron2.utils.logger import setup_logger
                # setup_logger()
                
                # # import some common libraries
                # import numpy as np
                # import cv2
                # import random

                # # import some common detectron2 utilities
                # from detectron2 import model_zoo
                # from detectron2.engine import DefaultPredictor
                # from detectron2.config import get_cfg
                # from detectron2.utils.visualizer import Visualizer
                # from detectron2.data import MetadataCatalog

                # import matplotlib.pyplot as plt

                # im = cv2.imread("../img/test_img/white.png")
                

                # cfg = get_cfg()
                # cfg.merge_from_file(("../config_files/f-rcnn-config.yaml"))
                # cfg.MODEL.DEVICE = "cpu"
                # # Create predictor
                # predictor = DefaultPredictor(cfg)

                # Make prediction

                frcnn_thread = threading.Thread(target=predict_frcnn, args=(frame,))

                if not FRCNN_ACTIVITY:
                    frcnn_thread.start()

                # im = frame
                # outputs = predictor(im)
                # classes_result = (outputs["instances"].pred_classes).numpy()
                # for class_result in classes_result:
                #     if class_result == 0:
                #         winsound.Beep(250, 1000)

                # print(outputs)
                # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # plt.figure(figsize = (14, 10))
                # plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

                """End F-RCNN"""

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
        
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2_imshow("output", frame)
        # cv2.imshow(frame)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b' --frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        # if cv2.waitKey(1) > -1:
        #     print("finished by user")
        #     break



INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

model_path = "../models/best17.onnx"

# video_path = "/content/drive/MyDrive/colabenv/elephant attack.mp4"

class_list = load_classes()

is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"


net = build_model(is_cuda)
capture = load_capture()
print(capture)

start = time.time_ns()
fps = -1



print("Total frames: " + str(total_frames))

# capture.release()