# -*- coding: utf-8 -*-
"""
@author: lisha
modified code to work on any single image 
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import cv2
from cv2 import CascadeClassifier
import numpy as np
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

import tensorflow as tf

from CNNmodels import FAN_crf_pred



import scipy.io

#%%----------------------------------------------------------------


FLAGS = tf.app.flags.FLAGS

# configurations

tf.app.flags.DEFINE_integer("batch_size",
                    default=1,
                    help="Batch size.")
tf.app.flags.DEFINE_integer("gt_num_lmks",
                    default=68,
                    help="Number of landmarks in ground truth")
tf.app.flags.DEFINE_integer("num_lmks",
                    default=68,
                    help="Number of landmarks in prediction")
tf.app.flags.DEFINE_integer("eval_num",
                    default=689,
                    help="Number of evaluation faces.")
tf.app.flags.DEFINE_float("offset",
                    default=0.,
                    help="Offset to add to prediction.")
# directories

tf.app.flags.DEFINE_string('data_eval_dir',
    './data/300w_train_val/val/',
    help=""" eval data folder""")
tf.app.flags.DEFINE_string('eval_tfrecords_file',
    'thrWtrain_val_689.tfrecords',
    help=""" eval tfrecords file""")

tf.app.flags.DEFINE_string('model_dir',
    './pretrained_models/300wtrain/',
    """Directory for model file""")
tf.app.flags.DEFINE_string('model_name',
    'model_300wtrain.ckpt',
    """model file name""")
tf.app.flags.DEFINE_string('facemodel_path',
    './facemodel/DM68_wild34.mat',
    # './facemodel/DM68_lp34.mat',
    """face model path""")
tf.app.flags.DEFINE_string('save_result_name',
    '/results/image.mat',
    """mat file path to save result""")
#%%----------------------------------------------------------------


def main(argv=None):

    model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
    FAN_crf_model = FAN_crf_pred.FAN_crf_eval(model_path = model_path, FLAGS=FLAGS)

    outdir='../results/face/'
    os.mkdir('../results/cnn_crf/')
    filename=FLAGS.subj_filename
    print("Processing ", filename)
    coord=open(filename,"r");


    Lines=coord.readlines()
    for line in Lines:
        subj=line.rstrip('\n')
        basename = os.path.basename(subj)
        print(subj+'_both.wmv')
        vidcap = cv2.VideoCapture(subj+'_both.wmv')
    
        face_file= outdir+basename+'_face_coord.npy'                       
        if (not os.path.exists(face_file)):
            face_file=outdir+basename+'_both_face_coord.npy'

        hasFrames,image = vidcap.read()
        print("========================> " ,hasFrames, np.shape(image), vidcap.isOpened())
        predlist=[]
        while(hasFrames):
            if hasFrames:
                fno=vidcap.get(cv2.CAP_PROP_POS_FRAMES)
                filename=os.path.basename(subj)
                im_str=subj+str(fno+1)
                rect=np.load(face_file)    
                y = max(rect[0]-70,0)
                y2 = max(rect[1]-50,0)
                x = rect[2]+80
                x2 = max(rect[3]-80,0)
                print("clipped: ",y, y2, x,x2) 
                cv2.imwrite('../results/cnn_crf/'+basename+"_cnn_crf.jpg", image[y:y2,x:x2,:])     # save frame as JPG file
   

                preds, precision, face = FAN_crf_model.predict_single(img_string= "../results/cnn_crf/"+filename+"_cnn_crf.jpg")
                predlist.append(preds)
            hasFrames,image = vidcap.read()
        print(np.shape(predlist))
        scipy.io.savemat('../results/cnn_crf/'+basename+'_cnn_crf.mat', 
                    {"joint_mean":predlist,
                    "inv_cov":precision, "face":face })

if __name__ == '__main__':
    filename = sys.argv[1]
    tf.app.flags.DEFINE_string('subj_filename',
        filename,
        help=""" eval data list file""")
    tf.app.run()









