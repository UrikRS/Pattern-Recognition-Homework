import cv2
import numpy as np
import matplotlib.pyplot as plt
from cmath import exp

sample_num = 3 # 製作模型時想要使用的樣本張數，ω0, ω1使用相同數量的樣本

def get_imgs(dir_name):
    """ 收集資料夾中的所有圖片

    Parameters
    ----------
    dir_name : string
        資料夾位置

    Returns
    -------
    imgs : list
        [img1, img2, img3, ...]
        img : ndarray
        樣本圖片 [[px11, px12, ...], [px21, px22, ...], ...]
        pixel : [BLUE, GREEN, RED]
    """
    imgs = []
    for i in range(sample_num):
        imgs.append(cv2.imread(dir_name + '\\%d.jpg'%i, flags=1))
    return imgs

def make_model(samples):
    """ 製造模組 : ωi

    計算各樣本的 mean vector, covariance matrix。

    Parameters
    ----------
    samples : list
        樣本圖片集 [img1, img2, img3, ...]

    Returns
    -------
    [mu, cov] : list
        [[μ1, μ2, ...], [Σ1, Σ2, ...]]
        μi : sample i 的 mean vector [BLUE_mean, GREEN_mean, RED_mean]
        Σi : sample i 的 covariance matrix [[σ11, σ12, σ13], [σ21, σ22, σ23], [σ31, σ32, σ33]]
    """
    mu, cov = [], []
    for sample in samples:
        mu.append(np.mean(sample.reshape(-1, 3), axis=0))
        cov.append(np.cov(sample.reshape(-1, 3).T))
    return [mu, cov]

def classif(x, w0, w1):
    """ 判斷像素是否屬於鴨的一部分

    以象素 x 對兩個模型中的每個 (μi, Σi) 計算 :

    P(x|ωi) = 1 / (√(2π^3) * √|Σi|) * exp(-1/2 * transform(x-μi) ⋅ inverse(Σi) ⋅ (x-μi))

    比較最大值 P(x|ω0)_max, P(x|ω1)_max，判斷 x 屬於 ω0 或 ω1。

    Parameters
    ----------
    x : array
        pixel [BLUE, GREEN, RED]
    w0 : list
        非鴨模型 ω0 [[μi], [Σi]]
    w1 : list
        鴨模型 ω1 [[μi], [Σi]]

    Returns
    -------
    BLACK or WHITE ? P(x|ω0)_max ≥ P(x|ω1)_max
    """
    p_x_w0_max, p_x_w1_max = [[0]*3]*3, [[0]*3]*3
    for i in range(sample_num):
        p_x_w0 = 1/(pow(2*np.pi, 3/2)*pow(abs(w0[1][i]), 1/2))*exp(-1/2*(x-w0[0][i]).T.dot(np.linalg.pinv(w0[1][i])).dot(x-w0[0][i]))
        p_x_w1 = 1/(pow(2*np.pi, 3/2)*pow(abs(w1[1][i]), 1/2))*exp(-1/2*(x-w1[0][i]).T.dot(np.linalg.pinv(w1[1][i])).dot(x-w1[0][i]))
        if (p_x_w0 > p_x_w0_max).any(): p_x_w0_max = p_x_w0
        if (p_x_w1 > p_x_w1_max).any(): p_x_w1_max = p_x_w1
    if (p_x_w0_max >= p_x_w1_max).any(): return [0, 0, 0]
    else: return [255, 255, 255]

def make_picture(data, w0, w1):
    """ 製造要輸出的圖片

    將受測圖片 reshape 成 2D array [pixel1, pixel2, ...]，

    將每個 pixel 使用 classif 的結果儲存於 list，

    將此 list 轉換為 array，reshape 成原本的結構回傳。

    Parameters
    ----------
    data : ndarray
        受測圖片 [[px11, px12, ...], [px21, px22, ...], ...]
        pixel : [BLUE, GREEN, RED]
    w0 : list
        非鴨模型 [[μi], [Σi]]
    w1 : list
        鴨模型 [[μi], [Σi]]

    Returns
    -------
    np.asarray(result_pic).reshape(old_shape) : ndarray
        可輸出的圖片
    """
    old_shape = data.shape
    data = data.reshape(-1, 3)
    result_pic = []
    i=0
    for d in data:
        i += 1
        result_pic.append(classif(d, w0, w1))
        print('running %.1f %%'%(i*100/np.size(data, axis=0))) # running means living
    return np.asarray(result_pic).reshape(old_shape)

if __name__ == '__main__':
    no_duck = get_imgs('Homework1\\pic\\not duck')
    ducks = get_imgs('Homework1\\pic\\duck')
    test_pic = cv2.imread('Homework1\\pic\\tiny.jpg', flags=1)
    w0 = make_model(no_duck)
    w1 = make_model(ducks)
    result_pic = make_picture(test_pic, w0, w1)
    plt.imshow(result_pic)
    plt.show()