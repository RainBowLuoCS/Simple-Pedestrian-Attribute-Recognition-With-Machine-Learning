import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib
import cv2
from config import *
from skimage import color
import torch
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import random
import os
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
id_set= {}
P=0.02
Path=root_path = os.path.abspath(os.path.dirname(__file__)).split('human-detector')[0]
upperbody_detector=cv2.CascadeClassifier(os.path.join(Path,'human-detector/data/cascadexml/haarcascade_upperbody.xml'))
lowerbody_detector=cv2.CascadeClassifier(os.path.join(Path,'human-detector/data/cascadexml/haarcascade_lowerbody.xml'))
nose_detector=cv2.CascadeClassifier(os.path.join(Path,'human-detector/data/cascadexml/haarcascade_mcs_nose.xml'))
eye_detector=cv2.CascadeClassifier(os.path.join(Path,'human-detector/data/cascadexml/haarcascade_eye.xml'))
color_space={
    'blue':[(100,100,50),(130,255,255)],
    'red':[(0,50,50),(10,255,255)],
    'green':[(35,45,45),(80,255,255)],
    'white':[(0,0,221),(180,30,255)],
    'black':[(0,0,0),(180,255,45)]
}
color_label=['blue','red','green','white','black']
def attribute_model(img):
    pp=img.copy()
    tt=cv2.cvtColor(pp,cv2.COLOR_BGR2GRAY)
    result={}
    result['mask']='no'
    result['glasses']='no'
    result['upper_clothes_color']='white'
    result['lower_clothes_color']='black'
    res=upperbody_detector.detectMultiScale(tt,1.2,3)
    if len(res)>0:
        x,y,w,h =res[0]
        src=img[y:y+h,x:x+w,:]
        hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        s=hsv.shape[0]*hsv.shape[1]
        rect= []
        for i in color_space.keys():
            mask=cv2.inRange(hsv,color_space[i][0],color_space[i][1])
            rect.append(mask.sum()/s)
        result['upper_clothes_color']=color_label[rect.index(max(rect))]
    res=lowerbody_detector.detectMultiScale(tt,1.2,3)
    if len(res)>0:
        x,y,w,h =res[0]
        src=img[y:y+h,x:x+w,:]
        hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        s=hsv.shape[0]*hsv.shape[1]
        rect= []
        for i in color_space.keys():
            mask=cv2.inRange(hsv,color_space[i][0],color_space[i][1])
            rect.append(mask.sum()/s)
        result['lower_clothes_color']=color_label[rect.index(max(rect))]
    res=nose_detector.detectMultiScale(tt,1.2,3)
    if len(res)!=0:
        result['mask']='yes'
    res=eye_detector.detectMultiScale(tt,1.2,3)
    if len(res)!=0:
        result['glasses']='yes'
    return result
def plot_bboxes(image, bboxes,indice_system, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        src=image[y1:y2,x1:x2,:]
        if pos_id in id_set.keys():
            seed=random.random()
            if seed>1-P:
                out=attribute_model(src)
                f = True
                for k in indice_system.keys():
                    if k in out.keys():
                        if out[k] != indice_system[k]:
                            f = False
                if f:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                id_set[pos_id] = color
            else:
                pass
        else:
            out=attribute_model(src)
            f=True
            for k in indice_system.keys():
                if k in out.keys():
                    if out[k]!=indice_system[k]:
                        f=False
                        break
            if f:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            id_set[pos_id]=color

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, id_set[pos_id], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, id_set[pos_id], -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(target_detector, image):
    new_faces = []
    bboxes,img = target_detector.detector(image)

    image=img

    bbox_xywh = []
    confs = []
    clss = []
    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw, target_detector.indice_system)

    return image, new_faces, face_bboxes

class Detector:
    def __init__(self,indice_system):
        self.indice_system=indice_system
        self.svm_clf = joblib.load(model_path)
        self.cas_clf = cv2.CascadeClassifier(
            os.path.join(Path,
            'human-detector\data\cascadexml\haarcascade_fullbody.xml')
        )
        self.deepsort=deepsort
        self.build_config()
    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def detector(self,im):

        # List to store the detections
        rects = []
        sc =[]
        # The current scale of the image

        clone = im.copy()
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        boxs=self.cas_clf.detectMultiScale(grey,scaleFactor=1.02,minNeighbors=3)
        if len(boxs) > 0:
            for x, y, w, h in boxs:
                # y -= (h // 8)
                # y = max(0, y)
                # h += (h // 8)
                # h = min(h, grey.shape[1])
                # cv2.imshow('aa', cv2.resize(grey[y:y + h, x:x + w], dsize=min_wdw_sz))
                # cv2.waitKey(0)
                fd = hog(cv2.resize(grey[y:y + h, x:x + w], dsize=min_wdw_sz), orientations, pixels_per_cell,
                         cells_per_block, visualize=visualize, block_norm='L1-sqrt')
                pred_prob = self.svm_clf.predict_proba(fd.reshape(1, -1))
                sc.append(pred_prob[0][1])
                rects.append([x, y, x+w, y+h])


        sc = np.array(sc)
        rects=np.array(rects)
        pick = non_max_suppression(rects, probs=sc, overlapThresh=0.2)

        rects = rects.tolist()
        res = []
        for i in pick:
            res.append(sc[rects.index(i.tolist())])
        result=[]
        for (xA, yA, xB, yB),i in zip(pick,res):
            result.append((xA, yA, xB, yB,'person',i))
        return result,clone

    def feedBa(self,img):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        im, faces, face_bboxes = update_tracker(self, img)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes

        return retDict



