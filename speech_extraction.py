
import speech_recognition
import pyttsx3
import os
recognizer = speech_recognition.Recognizer()
progpath1=os.path.dirname(os.path.realpath(__file__))
audpath1 = os.path.join(progpath1, 'voice')
os.chdir(audpath1)
b=''
def get(a):
	if a == 'otp':
		b = 'output1.wav'
	elif a == 'am':
		b = 'output2.wav'
	with speech_recognition.AudioFile(b) as source:
		audio_data = recognizer.record(source)
		text = recognizer.recognize_google(audio_data)
		#print(text)
	return text
