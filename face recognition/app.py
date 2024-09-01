from cgitb import text
from email.policy import default
from flask import Flask, render_template, Response, redirect,request,session,url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import csv
import os
from flask_session import Session
import pandas as pd
import numpy as np
import pytesseract
import face_recognition
import os   
from datetime import datetime
from flask import make_response
import pdfkit
from flask_mail import Mail, Message

app = Flask(__name__,template_folder='template')
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = "Huygens0p"



path = 'stud_img'
images = []
name=None
classNames = []
myList = os.listdir(path)
print(myList)
name_show=[]
for cl in myList:
  curImg = cv2.imread(f'{path}/{cl}')
  images.append(curImg)
  classNames.append(os.path.splitext(cl)[0])
  print(classNames)
CAMERA = None
CAMERA2=None
a=[]   
def gen_frames():
    global CAMERA
    global text
    global a
    CAMERA = cv2.VideoCapture(0)
    a=[]
    while True:
        pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract'

        success, frame = CAMERA.read()
        if not success:
            break
        else:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gaus=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,11)
            thr = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
            text=pytesseract.image_to_string(thr)
            print(text)
            if len(text) == 12:
               a.append(text)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            text= 'hello'
            # concat frame one by one and show result
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def findEncodings(images):
  encodeList = []
  for img in images:
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   encode = face_recognition.face_encodings(img)[0]
   encodeList.append(encode)
  return encodeList
encodeListKnown = findEncodings(images)




def gen_frames2():
    global CAMERA2
    CAMERA2 = cv2.VideoCapture(0)
    while True:
        success, frame = CAMERA2.read()
        if not success:
            break
        else:
           imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
           imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
           facesCurFrame = face_recognition.face_locations(imgS)
           encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
           for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                 matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                 faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                 matchIndex = np.argmin(faceDis)
 
                 if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
            # concat frame one by one and show result
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/open2')
def open2():
    message = 'open'
    #quizes=quiz.query.filter_by(qno=qno).first()
    return render_template('video.html',message=[message])

@app.route('/close2')
def close2():
    if CAMERA2 != None and CAMERA2.isOpened():
        CAMERA2.release()
    #CAMERA.release()
    cv2.destroyAllWindows()
    '''dict={'name': name}
    df=pd.DataFrame(dict)
    df.to_csv('Attendence.csv')
    #nameList.append(name)'''
    #quizes=quiz.query.filter_by(qno=qno).first()
    return render_template('video.html')


@app.route('/login',methods=['GET','POST'])
def login():
  if request.method=="POST":
       username=request.form['email']
       password=request.form['password']
       if(username=='vaibhavdixit384@gmail.com' and password=='1234'):
           session['email']=username
           return render_template('home.html',email=username)
  else:
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return render_template('login.html')

    


    


if __name__ == "__main__":
    app.run(debug=True, port=8000)
    app.run()