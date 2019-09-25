import cv2
import numpy as np
import os

def Thin(image,array):
    h,w = image.shape
    iThin = image
 
    for i in range(h):
        for j in range(w):
            if image[i,j] == 0:
                a = [1]*9
                for k in range(3):
                    for l in range(3):
                        #如果3*3矩阵的点不在边界且这些值为零，也就是黑色的点
                        if -1<(i-1+k)<h and -1<(j-1+l)<w and iThin[i-1+k,j-1+l]==0:
                            a[k*3+l] = 0
                sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                #然后根据array表，对ithin的那一点进行赋值。
                iThin[i,j] = array[sum]*255
    return iThin        
    
#最简单的二值化函数，阈值根据图片的昏暗程度自己设定，我选的180
def Two(image):
    w,h = image.shape
    size = (w,h)
    iTwo = image
    for i in range(w):
        for j in range(h):
            if image[i,j]<180:
                iTwo[i,j] = 0 
            else:
                iTwo[i,j] = 255
    return iTwo
 
#映射表
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

         
def getCnt(img):
    
    img = img[0:a.shape[0]-250,:]
    img = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
    #img = cv2.Canny(img,100,200)
    #img = Thin(img,array)
    #img = Thin(img,array)
    #cv2.imshow('aa',a)
    #cv2.waitKey(0)

    pa, ca, ha = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    x = []
    for j in ca :
            for k in j :
                x.append(k)
    return x

def draw(y): # y -> cnt
    img2 = np.zeros((500, 500), np.uint8) 
    print(len(y))
    
    point_size = 1
    color = (0, 0, 255) # BGR
    thickness = 4 # 可以爲 0 、4、8
    
    for point in y[::20]:
        #print(point)
        cv2.circle(img2, tuple(point[0]), 1, (255), -1)

    WINDOW_NAME = 'Image de Lena'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()

    # Display an image
    cv2.imshow(WINDOW_NAME,img2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
                
                
                
datapath = "./test/"
datas = os.listdir(datapath)



a = cv2.imread("C:/Users/gordon/Desktop/2WEEK/test_100307085-b0002.jpg",0)
x = getCnt(a)
    
for i in datas:
    #print("image :" + i +" :")#datapath + i +'/'+ i +'-b0002.jpg')
   
    b = cv2.imread(datapath + i +'/'+ i +'-b0002.jpg',0)
    y = getCnt(b)
    
    #print("image1 " +"contours : " + str(len(x)))
    print("image2 " + i)# +", contours : " + str(len(y)))
    
    #draw(y)

    if(len(y) >= 50000 or abs(len(x)-len(y)) > 5000):
        print("Contours difference > 5000,may be not matching")
        print('\n')
        continue
    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()
    
    #d1 = hd.computeDistance(np.asarray(x[::10]),np.asarray(y[::10]))
    d2 = sd.computeDistance(np.asarray(x[::10]),np.asarray(y[::10]))

    print("ShapeContextDistance :" + str(d2))
    print('\n')



