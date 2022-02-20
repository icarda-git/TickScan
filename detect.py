import argparse
import cv2
import matplotlib.pyplot as plt
import torch, torchvision
from PIL import Image
from IPython.display import Image, clear_output
import os

import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    # intialize the parser
    parser = argparse.ArgumentParser(
        description='Predict using tick detection models'
    )
    
    # arguments
    parser.add_argument('--model', help='Model to use for tick object detection [detr, ssd, fasterrcnn, yolov5, yolor, efficientdet]')
    parser.add_argument('--model_location', help='location of pretrained model')
    parser.add_argument('--input_location', help='location of input')
    parser.add_argument('--conf', help='confidence threshold', default=0.5)
    # parser.add_argument('--output_location', help='location of output')
    
    # Parse the arguments
    args = parser.parse_args()
    model = args.model
    model_location = args.model_location
    input_location = args.input_location
    conf = args.conf
    # output_location = args.output_location
    
            
            
    if model=='yolov5':
        os.chdir(os.getcwd()+'/yolov5')
        os.system(f'python detect.py --source {input_location} --weights {model_location} --conf-thres {conf} --exist-ok')

    else:
        print('Model does not exist')
    