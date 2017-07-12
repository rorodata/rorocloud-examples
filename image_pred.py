import numpy as np
import sys
import keras
import imagenet_labels 
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt


vgg16=keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

#b=plt.imread('/data/animal1.jpg')
b=plt.imread(sys.argv[1])

plt.imshow(b)

def topk(pred, k=3):
    topk=list(pred.argsort()[:,-k:].squeeze())
    topk.reverse()
    return topk

img=scipy.misc.imresize(b, (224,224))
pred=vgg16.predict(img.reshape(1,224,224,3))
top5=topk(pred,5)
for i in top5:
    print(pred[0,i], imagenet_labels.imgnet1000[i])

