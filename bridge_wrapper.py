'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True

import numpy as np
from numpy import sin, cos, tan
from scipy.spatial import distance

import cv2
import copy
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from model import vgg19
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image

def calc_K(fov_x, pixel_w, pixel_h, cx=None, cy=None):
    if cx is None:
        cx = pixel_w / 2.0
    if cy is None:
        cy = pixel_h / 2.0

    fx = 1.0 / (2.0 * tan(np.radians(fov_x) / 2.0)) * pixel_w
    fy = fx

    K = np.asarray([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
    ])

    return K

class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker

    # 推論用関数
    def inference(model, image):
        # 前処理
        device = torch.device('cpu')
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)

        # 推論
        with torch.set_grad_enabled(False):
            outputs, _ = model(input_image)

        # マップ取得
        result_map = outputs[0, 0].cpu().numpy()

        # マップから人数を計測
        count = torch.sum(outputs).item()

        return result_map, int(count)

    # マップ可視化用関数
    def create_color_map(result_map, image):
        color_map = copy.deepcopy(result_map)

        # 0～255の範囲に正規化して、疑似カラーを適用
        color_map = (color_map - color_map.min()) / (color_map.max() - color_map.min() + 1e-5)
        color_map = (color_map * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)

        # リサイズして元画像と合成
        image_width, image_height = image.shape[1], image.shape[0]
        color_map = cv2.resize(color_map, dsize=(image_width, image_height))
        debug_image = cv2.addWeighted(image, 0.35, color_map, 0.65, 1.0)

        return color_map, debug_image

    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
        cap = cv2.VideoCapture(video)
        # 横幅
        #print(f"width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        # 高さ
        #print(f"height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        videoPath = video
        cap = cv2.VideoCapture(videoPath)
        # カメラの視野角（水平方向）
        fov = 90
        # スクリーンの画素数（横）
        pw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # スクリーンの画素数（縦）
        ph = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # カメラ情報（内部パラメータ）
        cam_info = (fov, pw, ph)

        K = calc_K(*cam_info)

        # 訓練済み重みのパスを指定
        model_path = './model_nwpu.pth'

        # デバイスをCPUに指定
        device = torch.device('cpu')

        # モデルを構築
        model = vgg19()
        model.to(device)
        model.load_state_dict(torch.load(model_path, device))
        model.eval()
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        fig = plt.figure()
        ims = []
        group = []
        Group = []
        center_old = []
        not_group = []
        Not_group = []
        vector = []

        
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            result_map, count = self.inference(model, img)
            color_map, debug_image = self.create_color_map(result_map, img)
            cv2_imshow(debug_image)
            
            

        
        cv2.destroyAllWindows()
