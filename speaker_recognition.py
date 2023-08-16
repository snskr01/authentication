import os
import shutil
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from pathlib import Path
from IPython.display import display, Audio
from IPython.display import clear_output
# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
progpath=os.path.dirname(os.path.realpath(__file__))
DATASET_ROOT = os.path.join(progpath, "16000_pcm_speeches")
lol=os.path.dirname(os.path.realpath(__file__))
os.chdir(lol)
ACTUAL_PATH1 = os.path.join(lol, 'voice', 'output1.wav')
ACTUAL_PATH2 = os.path.join(lol, 'voice', 'output2.wav')


AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 100

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio

def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    
    audio = tf.squeeze(audio, axis=-1)
    #print(audio.shape())
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    fft = tf.expand_dims(fft, axis=0)
    

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    #print(fft)
    #a = fft.shape[1]
    #print(a)
    #print(tf.math.abs(fft[:, : (fft.shape[1] // 2), :]))
    return tf.math.abs(fft[:, : (fft.shape[1] // 2), :])

class_names = os.listdir(DATASET_AUDIO_PATH)
#print("Our class names: {}".format(class_names,))



def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

model=tf.keras.models.load_model('speaker.h5')
#print(model.summary())
ACTUAL_PATH = ''
def returnid(stri):
    if stri == 'otp':
        ACTUAL_PATH = ACTUAL_PATH1
    elif stri == 'am':
        ACTUAL_PATH = ACTUAL_PATH2
    voice=path_to_audio(ACTUAL_PATH)
    #voice = tf.squeeze(voice, axis=-1)
    #print(voice)
    #print('----------------------------------got audio----------------------------------------')
    fft=audio_to_fft(voice)
    #print('converted to fft')
    #print(fft)
    pred = model.predict(fft)
    #print(pred)
    names=os.listdir(DATASET_AUDIO_PATH)
    
    #print(names)
    #print('uid is ')
    result = np.where(pred == pred.max())
    #print(len(result))
    #print(names[(len(result))-1])
    return(names[(len(result))-1])
#print(returnid('am'))