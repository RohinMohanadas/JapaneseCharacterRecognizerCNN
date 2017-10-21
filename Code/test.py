import platform
import os
import numpy as np
#import scipy.misc
import dataimport as di
# from PIL import Image, ImageEnhance
# from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split


feature, labels = di.import_data('HIRAGANA')
print("done")
feature_arr = np.asarray(feature,dtype=np.float32)
labels_arr = np.asarray(labels,dtype=np.float32)
# labels_arr = labels_arr - 9250.0
# labels_arr = labels_arr - 166
print(len(feature_arr))
print("done with import")
# X_train, X_test, y_train, y_test = train_test_split(feature_arr,labels_arr,test_size=0.33, random_state=42)

#
# print(platform.system())
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + 'dataset/ETL1/ETL1C_01')
#
#
# mnist = learn.datasets.load_dataset("mnist")
# train_data = mnist.train.images  # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
# W,H = 28,28
# print(np.size(train_data))
# print(np.size(train_data[1]))
# for i in range(0,20):
#     scipy.misc.imsave('TestPics/outfile'+str(i)+'_'+str(train_labels[i])+'.png', np.reshape(train_data[i],(28,28)))
