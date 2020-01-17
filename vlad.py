import numpy as np
import cv2 as cv
import os
from sklearn.cluster import KMeans
import pickle
def siftDescribe():
    imgs_list = './tw_pic_crop/'
    imgs = os.listdir(imgs_list)
    #imgs = ['D157501-0007.png']
    sift_feat = []
    for img in imgs:
        print('\r'+img, end='')
        img = cv.imread(imgs_list + img)
        #gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(img,None)
        if des is None :
            continue
        sift_feat += [v for v in des]
        #img=cv.drawKeypoints(gray,kp,img)
        #print(len(des),des.shape)
        #cv.imwrite('sift_keypoints.jpg',img)
    
    print('sift is computed')
    np.save('feat',sift_feat)

def kmeanComupute():

    
    print('kmean is computing')
    feats = np.load('./feat.npy')
    
    kmeans = KMeans(n_clusters=16)
    kmeans.fit(feats)
    #print(kmeans.labels_)
    print('kean is computed')
    with open('kmeans.pickle','wb') as f:
        pickle.dump(kmeans,f)
    #print(kmeans.cluster_centers_[0],len(kmeans.cluster_centers_[0]))

def vlad():
    imgs_list = './tw_pic_crop/'
    feats_dir = './tw_pic_vlad/'
    imgs = os.listdir(imgs_list)
    with open('kmeans.pickle','rb') as f:
        kmeans=pickle.load(f)
    ct = kmeans.cluster_centers_ 
    k = kmeans.n_clusters
    for img in imgs:
        V = np.zeros([k,128])
        imgName = img
        print('\r' + img, end='')
        img = cv.imread(imgs_list + img)
        sift = cv.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(img,None)
        if des is None :
            continue
        
        predictLabels = kmeans.predict(des)
        for i in range(len(des)):
            V[predictLabels[i]] += des[i] - ct[predictLabels[i]]
        V = V.flatten()
        np.save(feats_dir + imgName[:-4], V)
            
def main():
    #siftDescribe()
    #kmeanComupute()
    vlad()
    
    
if __name__ == '__main__':
    main()
