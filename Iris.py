import urllib.request as req
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


#ファイルのダウンロード(アヤメデータ)
url = "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch2/iris/iris.csv"
savefile = "iris.csv"
req.urlretrieve(url, savefile)
iris_data = pd.read_csv(savefile, encording="utf-8")

y_labels = iris_data.loc[:,"name"]
x_data = iris_data.loc[:,
    ["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

#ラベルをOne-Hotベクトルに直す,基底を作る
labels ={
    'Iris-setosa': [1,0,0],
    'Iris-versicolor': [0,1,0],
    'Iris-virginica': [0,0,1]
}
y_nums = np.array(list(map(lambda v : labels[v], y_labels)))
x_data = np.array(x_data)

x_train, x_test, y_train, y_test= train_test_split(x_data, y_nums, train_size =0.8)

Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

#モデルの構築
model.compile(
    loss='categorical_crossentropy',
    optimizer = 'adam',
    metrics =['accuracy'])

#学習の実行
model.fit(x_train, y_train, batc_size=20, epochs=300)

#モデルの評価
score = model.evaluate(x_test, y_test, verbose=1)
print('正解率=', score[1], 'loss=', score[2])