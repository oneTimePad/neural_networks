import tensorflow as tf
from sklearn.metrics import accuracy_score
import mnist as  mnist_loader
import pdb
import numpy as np

"""
reference: Hands-On Machine Learning with Scikit-Learn and Tensorflow
"""

def numpy_to_tf(data):

    unzip = list(zip(*data))
    x = list(unzip[0])
    y = list(unzip[1])
    x = [t.ravel() for t in x]

    y = [(np.argmax(t) if not isinstance(t,np.int64) else t) for t in y]


    return x,y


train,valid,test = mnist_loader.load_data_wrapper()

X_train,y_train = numpy_to_tf(train)
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1,1)


X_test,y_test = numpy_to_tf(test)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[30],n_classes=10,\
                                        feature_columns=feature_columns)
dnn_clf.fit(x=X_train,y=y_train,batch_size=50,steps=40000)

acc = dnn_clf.evaluate(X_test,y_test)
print(acc)
