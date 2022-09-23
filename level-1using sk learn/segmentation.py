from matplotlib import image
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

def K_value_image_segmentation(n):
    k = n
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image2 = res.reshape((img_convert.shape))
    name="./images/image_k="+str(k)+".jpg"
    cv2.imwrite(name,result_image2)
    imag=Image.open(name)
    d=ImageDraw.Draw(imag)
    fnt=ImageFont.truetype("comicbd.ttf",25)
    text_to_write="k="+str(k)
    d.text((imag.size[0]/2,3),text_to_write,font=fnt,fill=(255,255,255))
    imag.save(name)

def gif_convertor_from_image():
    frames=[]
    for i in range(1,9):
        path="./images/image_k="+str(i)+".jpg"
        new_frame=Image.open(path)
        frames.append(new_frame)
    path="./images/image_k=15.jpg"    
    new_frame=Image.open(path)
    frames.append(new_frame)
    path="./images/image_k=25.jpg"    
    new_frame=Image.open(path)
    frames.append(new_frame)
    path="./images/image_k=45.jpg"    
    new_frame=Image.open(path)
    frames.append(new_frame)
    frames[0].save('./finalgif/k_values_gif.gif',format='GIF',append_images=frames[:14],save_all=True,duration=600,loop=100)

if __name__ == '__main__':
    img = cv2.imread("input_image.jpeg")
    img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    #defining image to experiment with number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    # reshape array to get the long list of RGB colors and then cluster using KMean()
    
    for i in range(1,11):
        K=i
        K_value_image_segmentation(K)

    K_value_image_segmentation(15)
    K_value_image_segmentation(25)
    K_value_image_segmentation(45)
    gif_convertor_from_image()