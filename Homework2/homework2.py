import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
    #print(data.shape)
    dimensions = [10, 20, 30, 40, 50]
    knn = KNeighborsClassifier(n_neighbors=3)
    for dime in dimensions:
        pca = PCA(n_components=dime, random_state=1234)
        x_train_t = pca.fit_transform(StandardScaler().fit_transform(x_train))
        x_test_t = pca.fit_transform(StandardScaler().fit_transform(x_test))
        knn.fit(x_train_t, y_train)
        print(knn.score(x_test_t, y_test))