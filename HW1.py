#Nazif ÇELİK 090200712
import numpy as np
from PIL import ImageGrab
from PIL import Image
import glob
import cv2
imagespath1 = glob.glob('C:/Users/n/PycharmProjects/pythonProject1/images/grayscale_car/*.jpg')
imagespath2 = glob.glob('C:/Users/n/PycharmProjects/pythonProject1/images/grayscale_car2/*.jpg')

    
while(True):
    try:
        for image1 in imagespath1:
            img1 =  cv2.imread(image1) 
            np_img1=np.array(img1)
            for image2 in imagespath2:
                img2 =  cv2.imread(image2) 
                np_img2=np.array(img2)
          
                sift = cv2.SIFT_create()
       
                kp1, des1 = sift.detectAndCompute(np_img1,None)
                kp2, des2 = sift.detectAndCompute(np_img2,None)
      
            FLANN_INDEX_KDTREE 	= 1
            index_params 		= dict(algorithm = 0, trees = 15)
            search_params 		= dict(checks=150)
            flann				= cv2.FlannBasedMatcher(index_params,search_params)
            matches 			= flann.knnMatch(np.float32(des1),np.float32(des2),k=2)

            matchesMask = [[0,0] for i in range(len(matches))]
            Olasilik 	= 0;

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    matchesMask[i]=[1,0]
                    pt1 = kp1[m.queryIdx].pt 
                    pt2 = kp2[n.trainIdx].pt 
                    Ust_Sag  = (int(pt1[0])-80,int(pt1[1])-80)
                    Alt_Sol  = (int(pt1[0])+80,int(pt1[1])+80)
               
                    Olasilik = Olasilik + 1
                    if(Olasilik > 5): 
                        j=j+1
                        cv2.rectangle(image1, Ust_Sag, Alt_Sol, (0,255,0),3)
                        Olasilik = 0
        draw_params = dict(
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)

        KNNCiz = cv2.drawMatchesKnn(np_img1,kp1,np_img2,kp2,matches,None,**draw_params)
        cv2.imshow('window',KNNCiz)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(e)
    pass

x= i
y= j
   
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train) 
