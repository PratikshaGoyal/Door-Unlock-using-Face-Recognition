#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import pyaudio
import os
from os import listdir
from os.path import isfile, join
import webbrowser
import time
import serial

import sendmail
import sms

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# Initialize Webcam
name = input("Enter your name :  ")
cap = cv2.VideoCapture(0)
count = 0

#create a directory for dataset of user
os.system(f"mkdir {name}" )
data_path = f"set the path of parent dir of name folder//{name}//"
path_to_dir = "path_to_parent_dir_of_name_dir//"
# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        cv2.imwrite(f'{data_path}//user' + str(count) + '.jpg', frame)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# Get the training data we previously made
# a=listdir('d:/faces')
# print(a)
# """
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i) 
    
# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)
model=cv2.face_LBPHFaceRecognizer.create()
# Initialize facial recognizer
# model = cv2.face_LBPHFaceRecognizer.create()
# model=cv2.f
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

#saving the model
model.write(f'{data_path}//trainer.yml')
print("Model saved sucessefully")


try:
    ser = serial.Serial('COM10',9600,timeout=0)
except:
    if not ser.isOpen():
        ser.open()

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model.read(f'{data_path}//trainer.yml')

def face_detector(img, size=0.5):    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

intr = 0
i= 0
while True:
    ret, frame = cap.read()    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        print(results)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        if confidence >= 90:
            cv2.putText(image, f"Hey {name}", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            sp = pyttsx3.init()
            sp.say(f"Welcome home {name}")
            sp.runAndWait()
            
            cv2.imshow('Face Recognition', image )
            #door unlock
            ser.write(b'v')
            time.sleep(10)
            break
        elif confidence < 90 and confidence >= 78:
            sp1 = pyttsx3.init()
            sp1.say("Sorry, your face can not be recognized. Do you want us to send an OTP to your phone ? ")
            sp1.runAndWait()
            mic  =  sr.Microphone()
            r = sr.Recognizer()
            with mic as source:
                audio = r.listen(source)
                text = r.recognize_google(audio)
            print(text)
            
            if text == 'YES' or text == 'Yes' or text == 'yes':
                print("hi")
                OTP = sms.sendSMS()
                time.sleep(5)
                while True:
                    try:
                        mic1  =  sr.Microphone()
                        r1 = sr.Recognizer()
                        sp = pyttsx3.init()
                        sp.say("Please speak the OTP sent on your phone")
                        sp.runAndWait()
                        with mic1 as source:
                            print("Please enter the OTP sent on your phone :")
                            audio = r1.listen(source)
                            text1 = r1.recognize_google(audio)
                        print(text1)

                        if text1 == OTP:
                            sp = pyttsx3.init()
                            sp.say(f"Welcome home {name}")
                            sp.runAndWait()
                            break
                        else:
                            sp = pyttsx3.init()
                            sp.say("Incorrect OTP")
                            sp.runAndWait()

                    except:
                        sp1 = pyttsx3.init()
                        sp1.say("Sorry, we are unable to perform the operations..Please wait..")
                        sp1.runAndWait()
                break
            else:
                cv2.imwrite(f"{path_to_dir}//intruder//intruder" + str(intr) + '.jpg',frame)
                sendmail.SendMail(f"{path_to_dir}//intruder//intruder" + str(intr) + '.jpg')                
                cv2.imshow('Face Recognition', image ) 
                time.sleep(40)
                imapper = easyimap.connect('imap.gmail.com', From, Password)
                for mail_id in imapper.listids(limit=1):
                    mail = imapper.mail(mail_id)
#                     print("hi")
                    print(mail.body[0:3])
                    if mail.body[0:3] == "yes" or mail.body[0:3] == "Yes":
                        #door unlock
                        ser.write(b'v')
                        time.sleep(10)
                break
            
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            #cv2.imshow('Face Recognition', image )
            intr += 1
            cv2.imwrite(f"{path_to_dir}//intruder//intruder" + str(intr) + '.jpg',image)
            SendMail(f"{path_to_dir}//intruder//intruder" + str(intr) + '.jpg')
            break

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
#         intr += 1
#         cv2.imwrite("C://Users//Dell//Desktop//intruder//intruder" + str(intr) + '.jpg',frame)
#         SendMail("C://Users//Dell//Desktop//intruder//intruder" + str(intr) + '.jpg')
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()   

