# -*- coding: utf-8 -*-
"""
Created on 2021

@author: lr
"""

from sklearn.svm import SVC
import joblib
import glob
import os
from config import *
import numpy as np

def train_svm():
    pos_feat_path = pos_feat_ph
    neg_feat_path = neg_feat_ph

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print(np.array(fds).shape,len(labels))
    if clf_type == "LIN_SVM":
        clf = SVC(kernel='linear',probability=True)
        print("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))
        
#训练SVM并保存模型
train_svm()