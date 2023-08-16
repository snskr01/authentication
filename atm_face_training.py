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
for gpu in gpus: 
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
cur_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_path)
def mini(a, b):
    if a > b:
        return a
    else:
        return b
print(cur_path)
for directory in os.listdir(os.path.join(cur_path, 'data', 'positive')):
    POS_PATH = os.path.join(cur_path, 'data', 'positive', directory, '*')
    positive = tf.data.Dataset.list_files(POS_PATH).take(1)
positives = tf.data.Dataset.zip((positive, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(positive)))))

for directory in os.listdir(os.path.join(cur_path, 'data', 'positive')):
    POS_PATH = os.path.join(cur_path, 'data', 'positive', directory, '*')
    positive = tf.data.Dataset.list_files(POS_PATH)
    positive_s = positive.shuffle(1000, reshuffle_each_iteration=True)
    positive1 = tf.data.Dataset.zip((positive, positive_s, tf.data.Dataset.from_tensor_slices(tf.ones(len(positive)))))
    positives = positives.concatenate(positive1)
positives = positives.shuffle(30000, reshuffle_each_iteration=True)
#positives = positives.take(1000)
#for i in positives:
    #print(i)



for directory in os.listdir(os.path.join(cur_path, 'data', 'positive')):
    PER_PATH = os.path.join(cur_path, 'data', 'positive', directory, '*')
negative = tf.data.Dataset.list_files(PER_PATH).take(1)
negatives = tf.data.Dataset.zip((negative, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(negative)))))

for directory in os.listdir(os.path.join(cur_path, 'data', 'positive')):
    for directory1 in os.listdir(os.path.join(cur_path, 'data', 'positive')):
        if directory != directory1:
            PER_PATH = os.path.join(cur_path, 'data', 'positive', directory, '*')
            NPER_PATH = os.path.join(cur_path, 'data', 'positive', directory1, '*')
            person = tf.data.Dataset.list_files(PER_PATH)
            n_person = tf.data.Dataset.list_files(NPER_PATH)
            negative1 = tf.data.Dataset.zip((person, n_person, tf.data.Dataset.from_tensor_slices(tf.zeros(len(person)))))
            negative2 = tf.data.Dataset.zip((n_person, person, tf.data.Dataset.from_tensor_slices(tf.zeros(len(person)))))
            negatives = negatives.concatenate(negative1)
            negatives = negatives.concatenate(negative2)
print('negs_size_1')
print(len(negatives))



print('pos_size')
print(len(positives))
print('negs_size_2')
print(len(negatives))
negatives = negatives.shuffle(30000, reshuffle_each_iteration=True)
negatives = negatives.take(len(positives))
#for i in negatives:
    #print(i)




def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    print(file_path)
    if img.shape[1]==2160:
        img = img[840:840+2160,0:2160, :]
    if img.shape[1]==1080:
        img = img[420:420+1080,0:1080, :]
    img = img/255
    img = tf.image.resize(img, (100,100), antialias=True)
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

data = positives.concatenate(negatives)
data = data.shuffle(10000, reshuffle_each_iteration=True)

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
train_samples = train_data.as_numpy_iterator()
def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    c1 = Conv2D(16, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    
    c2 = Conv2D(32, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
     
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')
embedding = make_embedding()
class L1Dist(Layer):
    
    
    def __init__(self, **kwargs):
        super().__init__()
       
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
def make_siamese_model(): 
    
    
    input_image = Input(name='input_img', shape=(100,100,3))
    
     
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
     
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
siamese_model = make_siamese_model()
binary_cross_loss = tf.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:     
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    print(loss)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        r = Recall()
        p = Precision()
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
EPOCHS = 50

with tf.device("/device:GPU:0"):
    train(train_data, EPOCHS)
siamese_model.compile(optimizer=opt, loss=binary_cross_loss)
siamese_model.save('siamesemodelwithgpu.h5')
from tensorflow.keras.metrics import Precision, Recall
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
siamese_model = tf.keras.models.load_model('siamesemodelwithgpu.h5', 
                                   custom_objects={'L1Dist':L1Dist})
y_hat = siamese_model.predict([test_input, test_val])
print([1 if prediction > 0.9 else 0 for prediction in y_hat ])
print(y_true)
m = Recall()
 
m.update_state(y_true, y_hat)

m.result().numpy()
 
m = Precision()
 
m.update_state(y_true, y_hat)

m.result().numpy()


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())
print(siamese_model.summary())