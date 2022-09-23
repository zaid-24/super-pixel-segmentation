import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from PIL import Image,ImageDraw,ImageFont

class KMeans(object): 
   
    def dist_btw_2points(self, x, y): 
        x_index_sumsquare = np.sum(np.square(x), axis=1)
        y_index_sumsquare = np.sum(np.square(y), axis=1)
        multiplication = np.dot(x, y.T)
        dists = np.sqrt(abs(x_index_sumsquare[:, np.newaxis] + y_index_sumsquare-2*multiplication))
        return dists

    def find_centers(self, points, K):
        row, col = points.shape
        retArr = np.empty([K, col])
        for number in range(K):
            randIndex = np.random.randint(row)
            retArr[number] = points[randIndex]
        return retArr

    def _update_centers(self, centers, points):
        row, col = points.shape
        global cluster_group
        cluster_group = np.empty([row])
        distances = self.dist_btw_2points(points, centers)
        cluster_group = np.argmin(distances, axis=1)
        K, D = centers.shape
        new_centers = np.empty(centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(points[cluster_group == i], axis=0)
        return new_centers

    def find_accuracy(self, centers, points): 
        dists = self.dist_btw_2points(points, centers)
        error = 0.0
        N, D = points.shape
        for i in range(N):
            error = error + np.square(dists[i][cluster_group[i]])
        return error

    def __call__(self, points,K):
        iterations_max = 100
        max_error1=1e-16
        maxerror2=1e-16
        centers = self.find_centers(points, K)
        for it in range(iterations_max):
            centers = self._update_centers(centers, points)
            error = self.find_accuracy(centers, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_error - error)
                if diff < max_error1 and diff / prev_error < maxerror2:
                    break
            prev_error = error
        return cluster_group, centers, error

def image_to_matrix(image_file, grays=False):
    img = plt.imread(image_file)
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img

def image_segmentation(n):
    image_matrix = image_to_matrix('input_image.png')
    row = image_matrix.shape[0]
    col = image_matrix.shape[1]
    x = image_matrix.shape[2]
    # flattening the image_matrix
    image_matrix = image_matrix.reshape(row*col,x)
    k = n 
    cluster_group, centers, error = KMeans()(image_matrix, k)
    image_updated = np.copy(image_matrix)
    for i in range(0, k):
        indices_current_cluster = np.where(cluster_group == i)[0]
        image_updated[indices_current_cluster] = centers[i]
    image_updated = image_updated.reshape(row, col, x)
    plt.figure(None, figsize=(25, 12))
    plt.axis('off')
    name="./images/image_k="+str(k)+".jpg"
    plt.imshow(image_updated)
    plt.savefig(name)
    plt.close()
    imag=Image.open(name)
    d=ImageDraw.Draw(imag)
    fnt=ImageFont.truetype("comicbd.ttf",75)    
    text_to_write="k="+str(k)
    d.text((imag.size[0]/2,150),text_to_write,font=fnt,fill=(255,255,255))
    imag.save(name)

def gif_convertor_from_image():
    frames=[]
    for i in range(1,11):
        if i==6:
            continue
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
    frames[0].save('./final_gif/k_values_gif.gif',format='GIF',append_images=frames[:14],save_all=True,duration=600,loop=100)

if __name__ == '__main__':
    for i in range(1,6):
        K=i
        image_segmentation(K)
    for i in range(7,11):
        K=i
        image_segmentation(K)
    image_segmentation(15)
    image_segmentation(25)
    image_segmentation(45)
    gif_convertor_from_image()