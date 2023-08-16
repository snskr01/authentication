THE READ ME FILE:


FILES TO RUN:
1. the driver code only if everything is set up.
2. atm face training or speaker recognition training for retraining the model
3. pac for converting the training audio file put in the folder to wav file of correct sampling rate
4. split for splitting the audio file to multiple segments for training
5. facecropper1 to crop faces out of photo dataset


THINGS TO NOTE:
1. sending the dataset would make the word file way bigger and it refused to open on some devices so we havent sent it, so instead we'll say how to make own dataset. the dataset we used will be sent if required to do so
2. the program needs a specific file directory structure to work, which will be given below (needed at least for dataset)
3. the program is made with consideration that the server or other computer running this has a powerful cpu, ssd and a powerful nvidia cuda enabled gpu with at least 6gb of vram on a 192bit bus. (program at least 50x slower without nvidia gpu)
4. executing the program on a computer without enough power WILL make it run slow.
5. active internet connection needed to send otp
6. the implementation of database is done via a dictionary. the dictionary is changed to add new uid and phone numbers (done in code)
7. some parts of the code do appear useless, but removing them causes issues, so they're still there. (eg: checks = checks)
8. tensorflow 2.11 onwards removes gpu acceleration for windows. The version used is 2.10.0 and is thus the recommended version to use
9. the dependencies to be installed (via pip if possible) are: (cv2, codecs, numpy, matplotlib, pyaudio, wave, os, scipy, noisereduce, pyttsx3, speech_recognition, shutil, pathlib, IPython, pydub, twilio.rest, time)     
10. put the zlibwapi file (included one) according to the instructions in https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#prerequisites-windows . we don't know why this isn't done during tensorflow installation because otherwise nothing works properly
11. cudnn and cuda installation is required (latest if possible)

FOLDER STRUCTURE:
root:
----16000_pcm_speeches:
--------audio:
------------(uid wise folders here for speaker database. eg: samples (all .wav) for uid 1000 in folder 1000. When audio is put here manually, execute pac.py and then split.py in order for that folder)
--------noise:
------------_background_noise_:
----------------(noise samples here : wav file at 16000hz sampling rate)
------------other:
----------------(noise samples here : wav file at 16000hz sampling rate)
--------tf_Wav_reader.py
----data:
--------current_faces:
------------captured:
--------------------(contains photos captured during execution, don't put anything here)
------------cropped:
--------------------(contains photos captured and cropped during execution, don't put anything here) 
--------positive:
------------(uid wise folders here for facial database. eg: samples (all .jpg) for uid 1000 in folder 1000. When photos are put in here manually, execute facecropper1.py again for that folder)
----video:
--------(contains video captured during execution, don't put anything here)
----audio:
--------(contains audio captured during execution, don't put anything here)
----(all the other python files and both the models here)