import numpy as np
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_pgm(dir_name):
    """ 讀取單張 .pgm 檔案

    Parameters
    ----------
    dir_name : string, 資料位址

    Returns
    -------
    img : array, 1D array
    """
    f = open(dir_name, 'rb')
    buffer = f.read()
    img = np.frombuffer(buffer, dtype='u1', count=10304, offset=14) #.reshape(112, 92)
    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.show() # want to see see what do they look like
    return img

def get_data(dir_name):
    """ 建立資料組
    
    Parameters
    ----------
    dir_name : string, 資料夾位址

    Returns
    -------
    x_train, y_train, x_test, y_test as array
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(1, 41):
        for j in range(1, 6):
            x_train.append(get_pgm(dir_name + '\\s%d\\%d.pgm'%(i,j)).reshape(10304))
            y_train.append(i)
    for i in range(1, 41):
        for j in range(6, 11):
            x_test.append(get_pgm(dir_name + '\\s%d\\%d.pgm'%(i,j)).reshape(10304))
            y_test.append(i)
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)

if __name__ == '__main__':
    x_train, y_train , x_test, y_test = get_data('Homework2\\att_faces')
    dimensions = [10, 20, 30, 40, 50]
    # standard scaler
    x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)

    for dime in dimensions:
        # apply PCA
        pca = PCA(n_components=dime, random_state=1234)
        rx_train, rx_test = pca.fit_transform(x_train), pca.transform(x_test)
        # apply KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(rx_train, y_train)
        predict = knn.predict(rx_test)
        # get report
        with np.printoptions(threshold=np.inf):
            f = open('%dD_PCA.txt'%dime, 'w')
            f.write(str(confusion_matrix(y_test, predict))+'\n'+classification_report(y_test, predict))
        # apply FLD
        lda = LinearDiscriminantAnalysis(n_components=dime, solver='eigen')
        tx_train, tx_test = lda.fit_transform(rx_train, y_train), lda.transform(rx_test)
        # apply KNN
        knn2 = KNeighborsClassifier(n_neighbors=3)
        knn2.fit(tx_train, y_train)
        predict2 = knn2.predict(tx_test)
        # get report
        with np.printoptions(threshold=np.inf):
            f2 = open('%dD_FLD.txt'%dime, 'w')
            f2.write(str(confusion_matrix(y_test, predict2))+'\n'+classification_report(y_test, predict2))