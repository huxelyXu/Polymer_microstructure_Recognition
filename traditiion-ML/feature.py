import cv2
import numpy as np

def feature(datapath, txtpath):
    imgs = []
    label = []
    label2 = []
    yuzhi = []
    tidu = []
    bianyuan = []
    pingjuntidu = []
    datainfo = open(txtpath, 'r')

    for line in datainfo:
        line = line.strip('\n')
        if line == 'train.bat' or line == 'train.txt':
            continue
        words = line.split('_')
        imgs.append(line)
        label.append(words[0])
        if words[0] == '0':
            label2.append('49.3')
        elif words[0] == '1':
            label2.append('51.2')
        elif words[0] == '2':
            label2.append('172.3')
        elif words[0] == '3':
            label2.append('149.4')
        elif words[0] == '4':
            label2.append('23.3')
    for i in imgs:
        dir_ = datapath + '/' + i
        img = cv2.imread(dir_, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        lei = i.split('_')
        if contours == [] or lei[0] == '4':
            cnt = np.zeros((24, 1, 2), dtype=int)
        else:
            # print(contours[-1])
            # print(i,lei)
            cnt = contours[-1]
        bianyuan_ = cv2.arcLength(cnt, True)
        bianyuan.append(bianyuan_)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        yuzhi_ = thresh1.sum()
        yuzhi.append(yuzhi_)
        kernel = np.ones((6, 6), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        tidu_ = gradient.sum()
        tidu.append(tidu_)
        if (bianyuan_ == 0):
            pingjuntidu.append(0)
        else:
            pingjuntidu.append(tidu_ / bianyuan_)

    return imgs, label, label2, yuzhi, tidu, bianyuan, pingjuntidu