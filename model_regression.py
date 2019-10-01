"""
Тренировка модели в режиме регрессии.
Точность модели после 50 эпох - 0.36
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import ReduceLROnPlateau
from datagenerators import DataGeneratorRegression


# Параметры модели
h = 28 # высота изображений
w = 28 # ширина изображений
c = 1 # количество каналов
seq_len = 10 # длина последовательности изображений
input_shape = (seq_len, h, w, c)
bs = 32 # batch size
num_classes = 1 # Количество классов, для регрессии - 1.

# Загрузка данных для тренировки и валидации
train_df = pd.read_csv("mnist_data/mnist_train.csv")
valid_df = pd.read_csv("mnist_data/mnist_test.csv")
train_data = DataGeneratorRegression(train_df, batch_size=bs,
				     dim=input_shape)
valid_data = DataGeneratorRegression(valid_df, batch_size=bs,
				     dim=input_shape)

# Количество классов, для регрессии - 1.
num_classes = 1

# Создание модели для регрессии
model = Sequential()
model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                     input_shape=input_shape,
                     padding='same', return_sequences=True))
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                     input_shape=input_shape,
                     padding='same', return_sequences=True))
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                     input_shape=input_shape,
                     padding='same', return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(num_classes))

model.compile('rmsprop', 'mse', metrics=['accuracy'])

# Callback для динамического изменения learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3,
			      verbose=1, factor=0.5, min_lr=1e-4)

model.fit_generator(generator=train_data, 
		    epochs=50, 
		    validation_data=valid_data,
		    callbacks=[reduce_lr])

model.save("models/model_regression.h5")

