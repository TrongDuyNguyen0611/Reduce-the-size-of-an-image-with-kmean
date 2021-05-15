import matplotlib.pyplot
import numpy
from sklearn.cluster import KMeans


img=matplotlib.pyplot.imread("a.jpeg")

width=img.shape[0]
height=img.shape[1]

img=img.reshape(width*height,3)

kmeans=KMeans(n_clusters=4).fit(img)
lables=kmeans.predict(img)
clusters=kmeans.cluster_centers_


img2=numpy.zeros_like(img)

for i in range(len(img2)):
    img2[i]=clusters[lables[i]]

img2=img2.reshape(width,height,3)

matplotlib.pyplot.imshow(img2)
matplotlib.pyplot.show()

