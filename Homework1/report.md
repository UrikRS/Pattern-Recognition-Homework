# Homework 1 : Finding Ducks
## How I do the assignment
### Step 1 : 收集樣本
- 用小畫家從 full_duck.jpg 中收集鴨、非鴨的樣本，分別存在 duck, not duck 兩個資料夾。
- 為了方便存取，使用數字為檔案命名。
### Step 2 : 寫主程式
- 寫一個function收集資料夾中的所有樣本
    - 用 cv2.imshow 測試
    - 用 type 確認資料型態
    - print 看看資料長什麼形狀
- 寫一個function建造模組
    - 用 np.mean 找 mean vector
    - 用 np.cov 找 covariance matrix
    - print 看看資料長什麼形狀
- 寫一個function應用模組
    - 代入 P(x|ωi) 的公式$$ P(x|ω_i) = {1 \over (2π)^{d/2}|Σ_i|^{1/2}}exp\bigl(-{1 \over 2}(x-μ_i)^TΣ_i^{-1}(x-μ_i)\bigr)$$
    - print 測試錯誤
- 寫一個function輸出圖片
- 寫註解
### Step 3 : 測試圖片
- full_duck.jpg 的原始圖片太大，我寫的程式太慢，導致結果跑不出來，以為程式出錯，所以決定分割出較小的測試用圖片。
- 快速測試時使用的圖片 : 
    ![](https://i.imgur.com/QUOSb7B.jpg)
- 若不滿意測試結果就調整樣本，直到得出可以接受的結果 : 
    ![](https://i.imgur.com/tqXbzg8.png)
    (樣本數 : duck=5, noduck=5)

## The results
### Area 1
![](https://i.imgur.com/8l3bhp0.jpg)
![](https://i.imgur.com/1iY8IrP.png)

(樣本數 : duck=5, noduck=5)

### Area 2
![](https://i.imgur.com/q8Ccanz.jpg)
![](https://i.imgur.com/xSDDyRN.png)

(樣本數 : duck=5, noduck=5)

### Area 3
![](https://i.imgur.com/xDYozRR.jpg)
![](https://i.imgur.com/Bj0fUEf.png)

(樣本數 : duck=5, noduck=5)

### Area 4
![](https://i.imgur.com/RhUWWWL.jpg)
![](https://i.imgur.com/sTr4yoc.png)

(樣本數 : duck=5, noduck=5)

### Area 5
![](https://i.imgur.com/gyTVp2q.jpg)
![](https://i.imgur.com/0dTjjNj.png)

(樣本數 : duck=5, noduck=5)

### full_duck.jpg
![](https://i.imgur.com/2VfrnR5.jpg)
![](https://i.imgur.com/KuAW7Ac.jpg)

(樣本數 : duck=3, noduck=3)


## Discussions
前後調整過很多次樣本，最後的結果還是有各種小石頭被誤認成鴨子，或者有鴨子被挖成空心的。而且只要總樣本數越大，運行時間就會倍數增加，當鴨、非鴨的樣本各取 5 個，總樣本共 10 個時，運行就已經相當吃力。

## Summary
從調整樣本的經驗中可以學到，因為鴨身上的顏色不常出現在環境色中，所以在選非鴨的樣本時可以先用肉眼將環境色大致分為幾類，各選其一，或者在同一張非鴨樣本裡將不同顏色各取一半，可以有效減少樣本數量。