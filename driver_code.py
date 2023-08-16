#1. walk in and person inputs user_id
#2. face gets captured via webcam in video and stored in video(capture yet to do)
#3. video to frame conversion and it is stored in data/current_faces/captured
#4. face extraction on captured and result is stored in data/current_faces/cropped and recognition is done on that. verified via atmface against input user_id
#5. they speak amount, the audio file is stored in audio folder via audiorecord.py
#6. speech extraction and speaker recognition on that audio file, output as text and boolean. speaker recognition code gets user_id which was verified before as input and outputs boolean
#7. balance is checked for the amount available from the database
#8. otp is calculated and sent to mobile of user_id. the value is also available to driver code
#9. its recorded and stored in audio. speech extract and speaker recognition is done and outputs ar text and boolean. otp is checked for in driver code.
#10. if otp is correct , transaction is done in the database and process successful
import os
uid=''
#taking uid as input
import capturevideo
import os
uid=input('input user id ')

print('face being captured')    


#print('after import')
capturevideo.captvid() #captures video

import photofromvideo
photofromvideo.pfv()

import face_cropper
face_cropper.fc()

import atm_face #yet to make from atm_face
uidface=atm_face.returnid(uid) #yet to change to only check for that particular id
#print(uidface)

if uid==uidface:
    print('face verified suceesfully')
else:
    print('second chance') 
    uidface=atm_face.returnid(uid)
    if uid==uidface:
        print('face verified successfully')
    else: 
        print('face verification failed')
        r=r'C:\Users\sansk\Desktop\My_Project_Files\5th_sem_projects\authentication\data\current_faces\cropped'
        w = os.listdir(r'C:\Users\sansk\Desktop\My_Project_Files\5th_sem_projects\authentication\data\current_faces\cropped')
        for fil in w:
            v = os.path.join(r,fil)
            os.remove(v)
        exit()
        
r=r'C:\Users\sansk\Desktop\My_Project_Files\5th_sem_projects\authentication\data\current_faces\cropped'
w = os.listdir(r'C:\Users\sansk\Desktop\My_Project_Files\5th_sem_projects\authentication\data\current_faces\cropped')
for fil in w:
    v = os.path.join(r,fil)
    os.remove(v)
    
print('say amount')
import audiorecorder
#print('after import')
import speaker_recognition
#print('after import')

#os.remove(r'C:\Users\sansk\Desktop\My_Project_Files\5th_sem_projects\authentication\voice\output.wav')
audiorecorder.aram() #record amount

import speech_extraction
amount = speech_extraction.get('am')
print('amount is ')
print(amount)
x = input('is input amount correct? (Y or N)')

if x=='Y':
    y=True
else:
    y=False
z=True
while z==True:
    uidvoice = speaker_recognition.returnid('am') #yet to be done, the particular function
    if y==False:
        audiorecorder.aram()
        amount = speech_extraction.get('am')
        print('amount is ')
        print(amount)
        x = input('is input amount correct? (Y or N)')
    
    if uid==uidvoice:
        print('voice verified successfully')
    else:
        print('second chance') 
        uidvoice=speaker_recognition.returnid('am')
        #print(uidvoice)
        if uid==uidvoice:
            print('voice verified successfully')
        else: 
            print('voice verification failed')
            exit()
    

    if x=='Y':
        z=False
    else:
        z=True
# input for asking whether or not the entered amount is correct over here
amount=float(amount)

import dbtransact #hrudhay to make
balance = dbtransact.getbalance(uid)#returns balance. uid is string
#print(balance)
transaction_happened=False

import otp_sender #hrudhay to send
number = dbtransact.getnumber(uid)
#print(number)
#otpsys = '1234'
#print(otpsys)
otpsys=otp_sender.motp(number)

print('wait 10 seconds for otp')
import time
time.sleep(10)
u = input('got otp message? (Y/N)')
print('say OTP')


audiorecorder.arotp() #OTP
# input for asking whether or not the entered amount is correct over here
otpvoice = speech_extraction.get('otp')
print('recorded OTP is ')
print(otpvoice)
x = input('is input OTP correct? (Y or N)')

if x=='Y':
    y=True
else:
    y=False
    exit()


if y==True:
    uidvoice = speaker_recognition.returnid('otp') #yet to be done, the particular function

    if uid==uidvoice:
        print('voice verified successfully')
        y=False
    else:
        print('second chance') 
        uidvoice=speaker_recognition.returnid('otp')
        #print(uidvoice)
        if uid==uidvoice:
            print('voice verified successfully')
            y=False
        else: 
            print('voice verification failed')
            exit()

    

if otpsys==otpvoice:
    print('correct OTP')
else:
    print('wrong OTP')
    exit()

if amount>balance:
    print('low balance for transaction')
else:
    rem_bal = dbtransact.withdraw(uid, amount) #amount is float, uid is string


print('successful withdraw')
print('remaining balance')
print(rem_bal)
