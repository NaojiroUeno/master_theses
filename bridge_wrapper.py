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


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0):
        # サンプル動画ファイル
        videoPath = video
        cap = cv2.VideoCapture(videoPath)
        # カメラの視野角（水平方向）
        fov = 50
        # スクリーンの画素数（横）
        pw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # スクリーンの画素数（縦）
        ph = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # カメラ情報（内部パラメータ）
        cam_info = (fov, pw, ph)

        K = calc_K(*cam_info)
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
        coord_old_list = []
        not_group = []
        Not_group = []
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

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            coord_set = []
            coord_list = []
            box_list = []
            center = []

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)
                tmp = [bbox[0], bbox[1], bbox[2], bbox[3], int(track.track_id)]
                box_list.append(tmp)

                x = (int(bbox[0]) + int(bbox[2])) / 2
                y = int(bbox[1])
                coord_set.append([x, y, 1, int(track.track_id)])

                cx = (int(bbox[0]) + int(bbox[2])) / 2
                cy = (int(bbox[1]) + int(bbox[3])) / 2
                tmp = [cx, cy, int(track.track_id)]
                center.append(tmp)


                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            for cs in coord_set:
                u, v, z, t_id = cs
                x = (u - K[0][2] * 1) / K[0][0] * 100
                y = (v - K[1][2] * 1) / K[1][1] * 100
                coord_list.append([x, y, t_id])
            
            if len(coord_list) > 0:
                x, y, person_id = zip(*coord_list)
                im = plt.plot(x, y, linestyle='None', marker='o', color="black")
                plt.gca().invert_yaxis()
                ims.append(im)

                # とりあえずここで座標間距離でグループを発見
                for i in range(len(coord_list) - 1):
                  for j in range(i + 1, len(coord_list)):
                      tmp1 = [coord_list[i][0], coord_list[i][1]]
                      tmp2 = [coord_list[j][0], coord_list[j][1]]
                      if distance.euclidean(tmp1, tmp2) < 4:
                        if len(group) == 0:
                          tmp = [coord_list[i][2], coord_list[j][2], 1]
                          group.append(tmp)
                          break
                        else:
                          flag = False
                          for k in range(len(group)):
                            if group[k][0] == coord_list[i][2]:
                              if group[k][1] == coord_list[j][2]:
                                flag = True
                                group[k][2] += 1
                                break
                    
                          if flag == False:
                            for k in range(len(group)):
                              if group[k][0] != coord_list[i][2]:
                                if group[k][1] != coord_list[j][2]:
                                  tmp = [coord_list[i][2], coord_list[j][2], 1]
                                  group.append(tmp)
                                  break
                                else:
                                  tmp = [coord_list[i][2], coord_list[j][2], 1]
                                  group.append(tmp)
                                  break
                              else:
                                tmp = [coord_list[i][2], coord_list[j][2], 1]
                                group.append(tmp)
                                break

                for l in range(len(group)):
                  if group[l][2] > 8:
                    flag = True
                    if len(Group) == 0:
                      tmp = [group[l][0], group[l][1]]
                      Group.append(tmp)
                      print("ID '" + str(group[l][0]) + "' and ID '" + str(group[l][1]) + "' are group!")
                      flag = False
                    else:
                      for i in range(len(Group)):
                        if group[l][0] == Group[i][0] and group[l][1] == Group[i][1]:
                          flag = False
                          break
              
                    if flag == True:
                      tmp = [group[l][0], group[l][1]]
                      Group.append(tmp)
                      print("ID '" + str(group[l][0]) + "' and ID '" + str(group[l][1]) + "' are group!")

                print(Group)

                

                ## 速度ベクトルを使用してグループ化どうかを判断 ##
                delta = []
                if frame_num % 5 == 0:
                  for i in range(len(center)):
                    for j in range(len(coord_old_list)):
                      if center[i][2] == coord_old_list[j][2]: # IDの照合
                        tmp = [center[i][0] - coord_old_list[j][0], center[i][1] - coord_old_list[j][1], center[i][2]]
                        delta.append(tmp)
                  print(delta)
                
                if len(delta) > 0:
                  for i in range(len(Group)):
                    for j in range(len(delta)):
                      if Group[i][0] == delta[j][2]: # Groupの1つ目のIDの照合
                        for k in range(j + 1, len(delta)):
                          if Group[i][1] == delta[k][2]: # Groupの2つ目のIDの照合
                            if abs(delta[j][0] - delta[k][0]) >= 7.0 or abs(delta[j][1] - delta[k][1]) >= 7.0:
                              if (delta[j][0] != 0.0 and delta[j][1] != 0.0) and (delta[k][0] != 0.0 and delta[k][1] != 0.0):
                                print("***************************************")
                                print("ID '" + str(Group[i][0]) + "' and ID '" + str(Group[i][1]) + "' aren't group!")
                                print("***************************************")
                                tmp = [Group[i][0], Group[i][1]]
                                flag = True
                                if len(not_group) == 0:
                                  not_group.append(tmp)
                                  flag = False
                                else:
                                  for x in range(len(not_group)):
                                    if tmp == not_group[x]:
                                      flag = False
                                      break
                                if flag == True:
                                  not_group.append(tmp)
                  
                for i in range(len(box_list) - 1):
                  for j in range(i + 1, len(box_list)):
                    for k in range(len(Group)):
                      if Group[k][0] == box_list[i][4] and Group[k][1] == box_list[j][4]:
                        if abs(box_list[i][0] - box_list[j][0]) > 50 or abs(box_list[i][1] - box_list[j][1]) > 50:
                          tmp = [Group[k][0], Group[k][1]]
                          not_group.append(tmp)
                
                ###### not_group内の重複を削除 ######
                Not_group = []
                for n_elem in not_group:
                  if not n_elem in Not_group:
                    Not_group.append(n_elem)

                ###### Group内からnot_groupに存在するペアを削除 #####
                new_group = []
                flag = False
                for i in range(len(Group)):
                  for j in range(len(Not_group)):
                    if Group[i] == Not_group[j]:
                      flag = True
                  if flag == False:
                    new_group.append(Group[i])
                  else:
                    flag = False

                if frame_num % 5 == 0:
                  print("------------------------------------")
                  print("group:  ", new_group)
                  Not_group = []
                  for n_elem in not_group:
                    if not n_elem in Not_group:
                      Not_group.append(n_elem)
                  print("not_group:  ", Not_group)
                  print("------------------------------------")
                  
                ########### ここにcv.rectangleでグループを囲む記述を書く ############
                flag_not = False
                for i in range(len(box_list) - 1):
                  for j in range(i + 1, len(box_list)):
                    for k in range(len(new_group)):
                      if new_group[k][0] == box_list[i][4] and new_group[k][1] == box_list[j][4]:
                          if int(box_list[i][0]) < int(box_list[j][0]): # box_list[i]がbox_list[j]より左にあるとき
                            if int(box_list[i][1]) > int(box_list[j][1]): # box_list[i]がbox_list[j]より下にあるとき
                              cv2.rectangle(frame, (int(box_list[i][0]), int(box_list[j][1])), (int(box_list[j][2]), int(box_list[i][3])), (0, 255, 255), 2)
                            else:
                              cv2.rectangle(frame, (int(box_list[i][0]), int(box_list[i][1])), (int(box_list[j][2]), int(box_list[j][3])), (0, 255, 255), 2)
                          else:
                            if int(box_list[i][1]) > int(box_list[j][1]): # box_list[i]がbox_list[j]より下にあるとき
                              cv2.rectangle(frame, (int(box_list[j][0]), int(box_list[j][1])), (int(box_list[i][2]), int(box_list[i][3])), (0, 255, 255), 2)
                            else:
                              cv2.rectangle(frame, (int(box_list[j][0]), int(box_list[i][1])), (int(box_list[i][2]), int(box_list[j][3])), (0, 255, 255), 2)
            
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: 
                  print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: 
                  print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if frame_num % 5 == 1:
              coord_old_list = center

        ani = animation.ArtistAnimation(fig, ims, interval=50)
        ani.save('anim.mp4', writer="ffmpeg")
        plt.show()

        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Group :  ", Group)
        print("Not Group : ", Not_group)
        
        cv2.destroyAllWindows()
