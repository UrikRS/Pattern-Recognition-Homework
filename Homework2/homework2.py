import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def get_pgm(dir_name):
    f = open(dir_name, 'rb')
    buffer = f.read()
    img = np.frombuffer(buffer, dtype='u1', count=10304, offset=14) #.reshape(112, 92)
    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.show() # want to see see what do they look like
    return img

def get_data(dir_name):
    data, label, test_d, test_l = [], [], [], []
    for i in range(1, 41):
        for j in range(1, 6):
            data.append(get_pgm(dir_name + '\\s%d\\%d.pgm'%(i,j)).reshape(10304))
            label.append(i)
    for i in range(1, 41):
        for j in range(6, 11):
            test_d.append(get_pgm(dir_name + '\\s%d\\%d.pgm'%(i,j)).reshape(10304))
            test_l.append(i)
    return np.asarray(data), np.asarray(label), np.asarray(test_d), np.asarray(test_l)

if __name__ == '__main__':
    x_train, y_train , x_test, y_test = get_data('Homework2\\att_faces')
    dimensions = [10, 20, 30, 40, 50]
    sx_train, sx_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
    for dime in dimensions:
        knn = KNeighborsClassifier(n_neighbors=3)
        pca = PCA(n_components=dime, random_state=1234)
        rx_train, rx_test = pca.fit_transform(sx_train), pca.transform(sx_test)
        knn.fit(rx_train, y_train)
        predict = knn.predict(rx_test)
        print(knn.score(rx_test, y_test))
        with np.printoptions(threshold=np.inf):
            f = open('%dD.txt'%dime, 'w')
            f.write(str(confusion_matrix(y_test, predict))+'\n'+classification_report(y_test, predict))