import numpy as np
import cv2
import os
#from matplotlib import pyplot as plt

#Import svm model
from sklearn import svm
import csv

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd

#Given a directory below function returns part of gray_img which is face alongwith its label/ID

X = np.array([])
#### For RGB color follow the two lines

Xfinal =np.ones(152128) # add length(X123) number of 1’s in Xfinal i.e. 152128
Yfinal=np.ones(1)

#directory='/home/sasmita/openface/retina_images/un effected/PATIENT 1'
directory='G://diabetic(g-7)//uneffected'
for path,subdirnames,filenames in os.walk(directory):
    for filename in filenames:
        if filename.startswith("."):    
            print("Skipping system file")  # Skipping files that startwith .
            continue

        id = os.path.basename(path)  # fetching subdirectory names
        img_path = os.path.join(path, filename)  # fetching image path
        test_img = cv2.imread(img_path)  # loading each image one by one
        if test_img is None:
            print("Image not loaded properly")
            continue

        img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) # Convert from cv's BRG default color order to RGB

        surf = cv2.xfeatures2d.SURF_create()
        keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)

        descriptors_surf1=descriptors_surf.flatten()
        print(descriptors_surf1.shape)

        descriptors_sift1=descriptors_sift.flatten()
        print(descriptors_sift1.shape)


        descriptors12=np.hstack([descriptors_surf1, descriptors_sift1])
        print(descriptors12.shape)
        print(len(descriptors12))
        L = len(descriptors12)
        if L<152128:
            N=152128-L
            ndescriptors12=np.pad(descriptors12, (0, N), 'constant')
        #X = np.append(X, L)

        Xfinal = np.vstack([Xfinal,ndescriptors12])
        Y=[0]
        Yfinal = np.vstack([Yfinal,Y])

directory='G://diabetic(g-7)//effected'
for path,subdirnames,filenames in os.walk(directory):
    for filename in filenames:
        if filename.startswith("."):
            print("Skipping system file")  # Skipping files that startwith .
            continue

        id = os.path.basename(path)  # fetching subdirectory names
        img_path = os.path.join(path, filename)  # fetching image path
        test_img = cv2.imread(img_path)  # loading each image one by one
        if test_img is None:
            print("Image not loaded properly")
            continue

        img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) # Convert from cv's BRG default color order to RGB

        surf = cv2.xfeatures2d.SURF_create()
        keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)

        descriptors_surf1=descriptors_surf.flatten()
        print(descriptors_surf1.shape)

        descriptors_sift1=descriptors_sift.flatten()
        print(descriptors_sift1.shape)


        descriptors12=np.hstack([descriptors_surf1, descriptors_sift1])
        print(descriptors12.shape)
        print(len(descriptors12))
        L = len(descriptors12)
        if L<152128:
            N=152128-L
            ndescriptors12=np.pad(descriptors12, (0, N), 'constant')
        #X = np.append(X, L)

        Xfinal = np.vstack([Xfinal,ndescriptors12])
        Y=[1]
        Yfinal = np.vstack([Yfinal,Y])
#print(X)
#print(np.max(X))
#print(ndescriptors12)
print(Xfinal)
print(Yfinal)

fields = ['File_Name', 'Actual']
with open('G://diabetic(g-7)//test_csv_file.csv','w') as csvFile:
    writer = csv.writer(csvFile)
    # writing the fields
    writer.writerow(fields)

csvFile.close()

file_names=[]
y_test=[]
X_test =np.ones(152128)
directory='G://diabetic(g-7)//test'
for path,subdirnames,filenames in os.walk(directory):
    for filename in filenames:
        if filename.startswith("."):
            print("Skipping system file")  # Skipping files that startwith .
            continue

        id = os.path.basename(path)  # fetching subdirectory names
        img_path = os.path.join(path, filename)  # fetching image path
        test_img = cv2.imread(img_path)  # loading each image one by one
        if test_img is None:
            print("Image not loaded properly")
            continue

        img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) # Convert from cv's BRG default color order to RGB

        surf = cv2.xfeatures2d.SURF_create()
        keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)

        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)

        descriptors_surf1=descriptors_surf.flatten()
        print(descriptors_surf1.shape)

        descriptors_sift1=descriptors_sift.flatten()
        print(descriptors_sift1.shape)


        descriptors12=np.hstack([descriptors_surf1, descriptors_sift1])
        print(descriptors12.shape)
        print(len(descriptors12))
        L = len(descriptors12)
        if L<152128: #168896
            N=152128-L
            ndescriptors12=np.pad(descriptors12, (0, N), 'constant')
        #X = np.append(X, L)

        with open('G://diabetic(g-7)//test_retina_img_details1.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([filename,filename[-5]])
            y_test.append(float(filename[-5]))

        csvFile.close()
        file_names.append(filename)
        X_test = np.vstack([X_test,ndescriptors12])
        #img_num +=1
        #Y=[1]
        #Yfinal = np.vstack([Yfinal,Y])


print(X_test)


####### Apply the SVM


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(Xfinal, Yfinal)
#ok
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred.shape) 
print(y_pred)
print(file_names)
print(y_test)

y_pred1=y_pred
#del y_pred1[0]
y_pred1=np.delete(y_pred1,0)

print(y_pred1.shape)
print("\ny_pred1 values\n",y_pred1)

#y_test will be found from manual excel sheet
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))

