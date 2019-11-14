# HOGH implement

# 算質心然後從質心切4份 
def slice(img): 

    centerX = 0
    centerY = 0
    centerCnt = 0
    
    # 累加X,Y PTS的位置
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x][y] > 0:
               centerCnt += 1
               centerX += x
               centerY += y

    # 除以 PTS 數 ，若圖上已沒有點，回傳NONE

    if centerCnt != 0 :
        centerX //= centerCnt
        centerY //= centerCnt
        s0 = img[:centerX,:centerY]
        s1 = img[centerX:,:centerY]
        s2 = img[:centerX,centerY:]
        s3 = img[centerX:,centerY:]

        
        return [s0,s1,s2,s3]
    
    return None,None,None,None
        
        

# 將梯度權重值轉換至直方圖上
def toHist(D):
    hist = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    for i in D:
        for j in i:
            hist[j] +=1
    return hist      
    
   
def bfs(queue,FA,FB,img):

    level = 1

    
    # 切割4份
    s = slice(img)
        
    # 將4份放入佇列中    
    queue.append([s[0],level])
    queue.append([s[1],level])
    queue.append([s[2],level])
    queue.append([s[3],level])
    
    # L0 = 3 時執行演算法1
    while queue[0][1] < 3:
        
        
        if queue[0][0] is None:
            queue.pop(0)
            continue
        s = slice(queue[0][0])
        
        # 計算梯度得到第一個回傳值並轉換直方
        # h 為該圖的特徵向量
        D = countD(queue[0][0])[0]
        h = toHist(D)
        
        
        # 把h特徵放入FA做最後的特徵值
        FA += [i for i in h]
        
        # 將分割的4份放入佇列中 LEVLE+1
        queue.append([s[0],queue[0][1]+1])
        queue.append([s[1],queue[0][1]+1])
        queue.append([s[2],queue[0][1]+1])
        queue.append([s[3],queue[0][1]+1])
        
        # POP目前的圖
        queue.pop(0)

    last_level = 3
    
    # L0 > 3 開始執行演算法
    
    # T為LEVEL特徵
    T = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while queue[0][1] <= 9:
        
               
        now_level = queue[0][1]
        
        # 若該圖回傳不正常則忽略
        if queue[0][0] is None:
            queue.pop(0)
            continue
            
        s = slice(queue[0][0])
        
        # 計算該圖梯度取第二個回傳值
        P = countD(queue[0][0])[1]
        
        # 該圖的梯度變化量為0 也忽略
        if sum(P) == 0:
            queue.pop(0)
            continue
        
        # 計算該圖的梯度直方取平均做二值化
        th = np.mean(P)
        P = [0 if i < th else 1 for i in P ]
        
        # 重新計算該圖的梯度加權直方
        Q = 0
        for i in P :
            x = 0
            Q += i * 2**x
            x +=1
        
        # 該圖直方向量加入到LEVEL的特徵向量 
        T[Q] +=1
        
        # 將同一LEVEL的特徵都合併在一個特徵向量
        # LEVEL 發生變化時重置 T LEVLE特徵
        if last_level != now_level:
            FB += [i for i in T]
            T = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            last_level = now_level
        
        # 將切割圖加入佇列
        queue.append([s[0],queue[0][1]+1])
        queue.append([s[1],queue[0][1]+1])
        queue.append([s[2],queue[0][1]+1])
        queue.append([s[3],queue[0][1]+1])
        

        
        queue.pop(0)
    
    # 把每一層LEVEL特徵加入至最終特徵值
    FA += [i for i in FB]
    
    return FA
    #print()
    
horizon = [[1,-1]]
verticle = [[1],[-1]]
diag = [[1,0],[0,-1]]
counter = [[0,1],[-1,0]]

# 計算梯度並對每一個pixel加權分數
# 兩個回傳值分別是小於L0用，大於L0用
def countD(img):

    # D計算每個pixel的梯度加權
    D = []
    # D2計算4種向量的數量
    D2 = [0,0,0,0]
    
    # 分別計算4種方向的梯度
    for x in range(len(img)):
        Dy = []
        for y in range(len(img[0])):
            G = 0
            # 直向
            if x+1 < len(img) and img[x+1][y] != img[x][y]:
                G += 1
                D2[0] +=1
            # 橫向
            if y+1 < len(img[0]) and img[x][y+1] != img[x][y]:
                G += 2
                D2[1] +=1
            # / 方向
            if x+1 < len(img) and y-1 >= 0 and img[x+1][y-1] != img[x][y]:
                G += 4
                D2[2] +=1
            # \ 方向
            if x+1 < len(img) and y+1 < len(img[0]) and img[x+1][y+1] != img[x][y]:
                G += 8
                D2[3] +=1
            Dy.append(G)
        D.append(Dy)

    return np.array(D),np.array(D2)


import cv2
import numpy as np
import os
import warnings
import time
import threading

warnings.filterwarnings('ignore')

zeroHist = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])    
        
if __name__ == '__main__':
    
    pic_path = './tw_pic_crop/'
    feature_path = './tw_pic_feature/'
        
    pics = os.listdir(pic_path)
    feature = os.listdir(feature_path)
    
    for pic in pics[::-1]:
        print('thread 1 :',pic)
        if pic[:-4]+'.npy' in feature:
            print('has be done')
            continue
        img = cv2.imread(pic_path + pic , cv2.IMREAD_GRAYSCALE)
        ret , img = cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
        print('loaded')
        tStart = time.time()
        F = bfs([],[],[],img)
        tEnd = time.time()
        print('compute time :',tEnd - tStart)
        if F is not None :
            np.save(feature_path+pic[:-4],F)


    
    print('done')

            







