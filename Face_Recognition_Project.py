#import module from tkinter for UI

from tkinter import *
import os
from datetime import datetime;
import cv2
import sys
from PIL import Image
import numpy as np
import pickle
import csv
import pandas as pd

#creating instance of TK
root=Tk()

#sys.path.append('C:\python 3.7\lib\site-packages')

root.configure(background="white")

#path=r"C:\Users\HP"
#os.chdir(path)

#root.geometry("300x300")

def function1():
    
    face_id=input('Enter your Id')
    face_name=input('Enter Your Name')

    if(__name__=="__main__"):
        

        # Creating individual folders for individual users #

        root="."  # informing the computer about the present directory
        folder="Python Train Images"
        name=face_name

        path=f"{root}/{folder}/{name}" 

        os.makedirs(path)

        # Folder of user Created #

        root1=r"C:\Users\Koushik Baagh\Desktop\Face Recognition Using Python\Face Recognition Using Python\Python Train Images"

        # Creating The specified Directory for cv2.imwrite()

        directory=f"{root1}\{name}"
        dirc=str(directory)

        # Creating Directory for .csv file

        root2=r"C:\Users\Koushik Baagh\Desktop\Face Recognition Using Python\Face Recognition Using Python"

        # Loading Haar CascadeClassifier
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


        capture=cv2.VideoCapture(0) # Opens Webcam

        count =0 #Initialize sample face image

        while(True):

            # Capture video frame
            ret, image_frame = capture.read()

            # Convert frame to grayscale
            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

            # Detect frames of different sizes, list of faces rectangles
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loops for each faces
            for (x,y,w,h) in faces:

                # Crop the image frame into rectangle
                cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

                # Increment sample face image
                count += 1

                roi_gray= gray[y:y+h, x:x+w] # Region Of Interest

                # Change the current working Directory for the image to be saved
                    
                os.chdir(dirc)
                       
                # Save the captured image into the training folder
                
                cv2.imwrite(str(face_name) + str(face_id) + '.' + str(count) + '.jpg',roi_gray)
                
                # cv2.imwrite("Python Train Images/"+ str(face_name) + str(face_id) + '.' + str(count) + '.jpg',roi_gray)
                # cv2.imwrite("Folder Location with this /" + name + id + number + Region of interest)

                # Display the video frame, with bounded rectangle on the person's face
                cv2.imshow('frame', image_frame)

            # To stop taking video, press 'q' for at least 100ms
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            # If image taken reach 100, stop taking video
            elif count>=51:
                print("Successfully Captured")
                break

        # Stop video
        capture.release()

        # Close all started windows
        cv2.destroyAllWindows()
        
    #########################
        os.chdir(root2)
        with open('StudentDetails.csv','a',newline='') as f: # The csv file is opened in append mode and [newline=''] is used to remove blank rows in between csv file...
            
            df=pd.DataFrame({'ID' : face_id,
                            'Name':pd.Categorical(face_name)})
            
            csvpath=r"C:\Users\Koushik Baagh\Desktop\Face Recognition Using Python\Face Recognition Using Python\StudentDetails.csv"
            if os.path.getsize(csvpath) == 0:
                df.to_csv(f, index=False)                 
            else:
                df.to_csv(f, header=False, index=False)
            

    
def function2():
    
    basedir=os.path.dirname(os.path.abspath('_file_')) # If I Put _file_ without invited commas then their will an errror
    imagedir=os.path.join(basedir,"Python Train Images")

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer= cv2.face.LBPHFaceRecognizer_create()

    faceSamples=[]
    y_labels=[]

    current_id=0
    label_ids={}

    for  root,dirs,files in os.walk(imagedir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG") or file.endswith("jpeg") or file.endswith("JPEG"):
                path=os.path.join(root,file)
                
                # What is the use of Label??? Label is used to extract the folder name of the captured images... 
                # which will be further used in Recognizer.py file
                
                label=os.path.basename(root).replace(" ","-").lower() # We can also write label=os.path.basename(os.path.dirname(path))
                #print(label, path)
                
                if not label in label_ids:       # 
                    label_ids[label]=current_id  # creating a label id for motherchod labels
                    current_id += 1              #
                    
                id_=label_ids[label]   # 'id' is a built in function in python so use 'id_'
                #print(label_ids)
                pil_image= Image.open(path).convert("L") # Grayscale
                size=(550, 550)
                final_image= pil_image.resize(size, Image.ANTIALIAS)

                
                #image=pil_image.rotate(45).show() Thois function is used to rotate image by 45 degrees
                #print(pil_image)
                
                image_array=np.array(pil_image,'uint8') # Turning this into numpy array "uint" stands for unsigned int
                #print(image_array)
                
                faces=face_cascade.detectMultiScale(image_array,1.32,5)
                
                for (x,y,w,h) in faces:
                    faceSamples.append(image_array[y:y+h,x:x+w])
                    y_labels.append(id_)

    # Using Pickle to save Label Ids                

    with open("labels.pickle",'wb') as f:
        pickle.dump(label_ids,f)

    # Train the OpenCV Recognizer

    recognizer.train(faceSamples,np.array(y_labels))
    recognizer.save("trainer.yml")
    print("Successfully trained")        
     
                    
                    
                    
    ###############################  PERFORMING FACE RECOGNITION ################################


    labels={}
    with open("labels.pickle",'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer= cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    capture=cv2.VideoCapture(0)


    font=cv2.FONT_HERSHEY_SIMPLEX
    name=labels[id_]
    color=(255,255,255)
    stroke=2

    while(True):

        # Capture video frame
        ret, image_frame = capture.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

        # Loops for each faces
        for (x,y,w,h) in faces:
            
            roi_gray = gray[y:y+h, x:x+w]

            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            id_,conf=recognizer.predict(roi_gray)
            if conf>= 45 and conf<=85:
                
                cv2.putText(image_frame, (labels[id_]) , (x,y) , font , 1 , color , stroke , cv2.LINE_AA )
                #print(labels[id_])
                
            else:
                str_="Intruder!!!Cannot Recognize"
                cv2.putText(image_frame, str_ , (x,y) , font , 1 , color , stroke , cv2.LINE_AA )
            
                
                
                
                    
               
                    
        cv2.imshow('frame',image_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()

                    
def function3():
    
    root.destroy()
    
#Starting title for the window
root.title("AUTOMATIC FACE RECOGNITION")

#creating a text label
Label(root, text="FACIAL RECOGNITION SYSTEM",font=("times new roman",20),fg="white",bg="maroon",height=2).grid(row=0,rowspan=2,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

#creating first button
Button(root,text="Take Images For Creating Dataset",font=("times new roman",20),bg="#0D47A1",fg='white',command=function1).grid(row=3,columnspan=2,sticky=W+E+N+S,padx=5,pady=5)

#creating second button
Button(root,text="Train + Recognize Dataset",font=("times new roman",20),bg="#0D47A1",fg='white',command=function2).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

#creating EXIT button
Button(root,text="Exit",font=('times new roman',20),bg="maroon",fg="white",command=function3).grid(row=9,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)


root.mainloop()
