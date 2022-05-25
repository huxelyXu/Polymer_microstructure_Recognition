import cv2
import os

path = 'E:/dataset/pic/'
filename = 'c.jpg'
cols = 64
rows = 64
img = cv2.imread(path+filename,1)
dim = (512, 512)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

sum_rows=img.shape[0]
sum_cols=img.shape[1]
m = 0
save_path='E:/dataset/pic/result/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols/cols),int(sum_rows/rows)))
for i in range(int(sum_cols/cols)):
    for j in range(int(sum_rows/rows)):
        m +=1
        cv2.imwrite(save_path+os.path.splitext(filename)[0]+"_"+str(m)+os.path.splitext(filename)[1]
                    ,img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:])
        #print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
print("裁剪完成，得到{0}张图片.".format(m))
print("文件保存在{0}".format(save_path))