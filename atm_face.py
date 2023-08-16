import cv2
import codecs
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#import tensorflow.compat.v1 as tf1
#tf1.disable_v2_behavior()
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
gpus = tf.config.experimental.list_physical_devices('GPU')
from keras.metrics import Precision, Recall
#for gpu in gpus: 
    #print(gpu)
    #tf.config.experimental.set_memory_growth(gpu, True)
progpath=os.path.dirname(os.path.realpath(__file__))
os.chdir(progpath)
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = img/255
    img = tf.image.resize(img, (100,100), antialias=True)
    return img

def preprocess_twin(input_img, validation_img):
    return(preprocess(input_img), preprocess(validation_img))

class L1Dist(Layer):
    
    
    def __init__(self, **kwargs):
        super().__init__()
       
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

siamese_model = tf.keras.models.load_model('siamesemodelwithgpu.h5', 
                                   custom_objects={'L1Dist':L1Dist})




#captfaces = captfaces.map(preprocess)
hits = {}
pairings = ()
def returnid(uid):
    captface = os.path.join(progpath, "data", "current_faces", "cropped", '*')
    valface = os.path.join(progpath, "data", "positive")
    captfaces = tf.data.Dataset.list_files(captface)
    #qa = 0
    #for q in os.listdir(captface):
        #wa = 0
        #for w in os.listdir(valface):
            #ra = 0
            #for r in os.listdir(os.path.join(valface, w)):
                #if ra>400:
                    #break
                #pairings[qa][wa][ra] = q , r
                #ra+=1
            #wa+=1
        #qa+=1
    #uids = os.listdir(valface)
    for name in os.listdir(valface):
        #print(os.listdir(valface))
        captfaces = tf.data.Dataset.list_files(captface)
        b = len(captfaces)
        #captfaces = captfaces.take(20)
        #print(captfaces)
        nameface = os.path.join(valface, name, '*')
        namefaces = tf.data.Dataset.list_files(nameface)
        a = len(namefaces)
        namefaces = namefaces.shuffle(3000, reshuffle_each_iteration=True)
        captfaces = captfaces.repeat(a)
        namefaces = namefaces.map(lambda x : tf.repeat(x, b))
        namefaces = namefaces.unbatch()
        #print(namefaces)
        
        #namefaces = namefaces.map(preprocess)
        
        checks = tf.data.Dataset.zip((namefaces, captfaces))
        checks = checks.shuffle(1000, reshuffle_each_iteration=True)
        e = len(checks)
        #print(e)
        checks = checks.map(preprocess_twin)
        checks = checks.batch(500)
        checks = checks.prefetch(50)
        #print(checks)
        u = checks.as_numpy_iterator()
        matches = 0
        for s in range(int(e/500)):
            namefaceins, captfaceins = u.next()
            
            #print(tf.shape(namefaceins))
            #print(tf.shape(captfaceins))
            #print(namefaceins.shape)
            
            i = [namefaceins, captfaceins]
            
            y_hat = siamese_model.predict(i)
            
            #print(y_hat)
            
            for prediction in y_hat:
                if prediction>0.8:
                    k=1
                else:
                    k=0
                matches = matches + k
        
        if name == uid:
            hits[name]=int(matches*2)
        else:
            hits[name]=int(matches/2)
        
    
    max = 0
    print(hits)
    for person,no in hits.items():
        if no > max:
            max = no
            detect = person
    #print("detected person")
    print(detect)
    #print("confidence")
    #print(max/400)
    #detect = detect
    return detect
#print(returnid())
#1. output should be user id
#2. it should get name from database using userid as primary key
#3. i should change name of folders to user_id