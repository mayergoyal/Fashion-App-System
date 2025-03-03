import pickle
import numpy as np
import tensorflow
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import cv2
feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False #already trained on imagenet

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() #adding layer
])
sampleinput=input("enter input")
img=image.load_img(sampleinput,target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
prepocessed_img=preprocess_input(expanded_img_array)
result=model.predict(prepocessed_img).flatten()
    #now need to normalise
normalized_res=result/ norm(result,ord=2)

#finding nearest neighbours
neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)
distances,indices=neighbors.kneighbors([normalized_res])
print(indices)

for file in indices[0]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)
