"""
Usage:
    $ python detect.py --model model_name --model_location model_path.pt --input_location image_path
    
model_name = [detr, ssd, fasterrcnn, yolov5, yolor, efficientdet]
if yolov5 you  are in app/yolov5/:
	model_path.pt & image_path should be relative paths


"""


import argparse
import cv2
import matplotlib.pyplot as plt
import torch, torchvision
from PIL import Image
from IPython.display import Image, clear_output

import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    # intialize the parser
    parser = argparse.ArgumentParser(
        description='Predict using tick detection models')
    # arguments
    parser.add_argument('--model', type=str, help='Model to use for tick object detection [detr, ssd, fasterrcnn, yolov5, yolor, efficientdet]')
    parser.add_argument('--model_location', type=str, help='location of pretrained model')
    parser.add_argument('--input_location', type=str, help='location of input image')
    # parser.add_argument('--output_location', help='location of output')
    
    # Parse the arguments
    args = parser.parse_args()
    model = args.model
    model_location = args.model_location
    input_location = args.input_location
    # output_location = args.output_location
    
    
    if model=='detr':
        first_class_index = 1
        assert(first_class_index in [0, 1])

        if first_class_index == 0:

          # There is one class, balloon, with ID n°0.

          num_classes = 11

          finetuned_classes = [
              'dromedarii_female', 'dromedarii_male', 'scupense_female', 'scupense_male', 'impeltatum_female', 'impeltatum_male', 'marginatum_female', 'marginatum_male', 'other_female', 'other_male', 'excavatum'
          ]

          # The `no_object` class will be automatically reserved by DETR with ID equal
          # to `num_classes`, so ID n°1 here.  

        else:

          # There is one class, balloon, with ID n°1.
          #
          # However, DETR assumes that indexing starts with 0, as in computer science,
          # so there is a dummy class with ID n°0.
          # Caveat: this dummy class is not the `no_object` class reserved by DETR.

          num_classes = 12

          finetuned_classes = [
              'N/A', 'dromedarii_female', 'dromedarii_male', 'scupense_female', 'scupense_male', 'impeltatum_female', 'impeltatum_male', 'marginatum_female', 'marginatum_male', 'other_female', 'other_male', 'excavatum'
          ]

          # The `no_object` class will be automatically reserved by DETR with ID equal
          # to `num_classes`, so ID n°2 here.

        print('First class index: {}'.format(first_class_index))  
        print('Parameter num_classes: {}'.format(num_classes))
        print('Fine-tuned classes: {}'.format(finetuned_classes))
        
        model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

        checkpoint = torch.load(model_location,
                        map_location='cpu')

        model.load_state_dict(checkpoint['model'],
                      strict=False)

        model.eval();
        
        img_name = input_location
        im = Image.open(img_name)

        run_worflow(im,
                    model)
            
    elif model=='ssd':
        CUSTOM_MODEL_NAME = 'ssd_mobnet_V2' 
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        
        paths = {
                'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
                'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
                'APIMODEL_PATH': os.path.join('Tensorflow','models'),
                'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
                'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('Tensorflow','protoc')
             }
             
        for path in paths.values():
            if not os.path.exists(path):
                if os.name == 'posix':
                	os.mkdir(path)
                if os.name == 'nt':
                    os.mkdir(path)
        
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
                
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(model_path, 'ckpt-41')).expect_partial()

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        
        import cv2 
        import numpy as np
        from matplotlib import pyplot as plt
        #%matplotlib inline
        
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        
        IMAGE_PATH = input_location
        
        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()
    
    elif model=='fasterrcnn':
        # Some basic setup:
        # Setup detectron2 logger
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()

        # import some common libraries
        import numpy as np
        import os, json, cv2, random
        from google.colab.patches import cv2_imshow

        # import some common detectron2 utilities
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog
    
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.02
        cfg.SOLVER.MAX_ITER = 4000   # 4000 iterations seems good enough, but you can certainly train longer
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # num of classes 
        cfg.MODEL.WEIGHTS = weights_path
        
        weights_path = model_location
        image_path = input_location
        config_file_path = "./trained_models/Faster_RCNN/faster_rcnn_R_50_FPN_3x.yaml"
        
        
        from detectron2.utils.visualizer import ColorMode

        for d in random.sample(dataset_dicts, 4):    
            im = cv2.imread(image_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=sample_metadata, 
                           scale=0.7, 
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2_imshow(v.get_image()[:, :, ::-1])
            
            
    elif model=='yolov5':
    	import os
    	#yolov5_path = 'icarda_yolov_zip/icarda_yolov/app/yolov5'
    	yolov5_path = 'app/yolov5'
    	os.chdir(yolov5_path)
    	os.system('python detect.py --source ' + input_location + ' --weights ' + model_location)
        
    elif model=='yolor':
    	import os
    	yolor_path = ''
    	os.chdi(yolor_path)
    	os.system('python detect.py --source '+ image_location +' --weights ' + model_location)
    
    elif model=='efficientdet':
        CUSTOM_MODEL_NAME = 'efficientdet' 
        PRETRAINED_MODEL_NAME = 'efficientdet_d1_coco17_tpu-32'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        
        paths = {
                'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
                'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
                'APIMODEL_PATH': os.path.join('Tensorflow','models'),
                'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
                'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('Tensorflow','protoc')
             }
             
        for path in paths.values():
          if not os.path.exists(path):
            if os.name == 'posix':
              os.mkdir(path)
            if os.name == 'nt':
              os.mkdir(path)
        
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
                
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(model_path, 'ckpt-41')).expect_partial()

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        
        import cv2 
        import numpy as np
        from matplotlib import pyplot as plt
        #%matplotlib inline
        
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        
        IMAGE_PATH = input_location
        
        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()

    else:
        print('Model does not exist')
    
