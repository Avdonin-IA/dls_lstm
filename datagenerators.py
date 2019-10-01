"""
Классы DataGenerator-ов для тренировки модели.
"""

import numpy as np
import pandas as pd
import keras


class DataGeneratorClassification(keras.utils.Sequence):
    """
    Генератор данных для изображений из MNIST.
    Формирует батчи в виде ряда следующей размерности: 
    (размер батча, количество изображений, высота, ширина).
    Метка - сумма первых двух чисел ряда.
    В качестве метки возвращается класс с использованием
    one-hot encoding.
    """
    def __init__(self, data_frame, batch_size=32, dim=(10, 28, 28, 1)):
        """
        Конструктор класса DataGenerator.
        Parameters:
            data_frame (pd.DataFrame): DataFrame, из которого формируются батчи;
            batch_size (int): Размер батча;
            dim (tuple): Размерность вида (Количество изображений, высота, ширина).
        """
        self.df = data_frame
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """ Перемешивает датасет """
        self.df = self.df.sample(frac=1)        
    
    def __len__(self):
        """ Возвращает количество батчей на одну эпоху """
        return int(np.floor(len(self.df)/(self.batch_size*self.dim[0])))
    
    def __getitem__(self, index):
        """ 
        Возвращает батч изображений
        Returns:
            X (np.array): Батч данных в формате (bs, seq_len, h, w, c),
                где bs - размер батча;
                seq_len - длина последовательности изображений;
                h - высота изображения;
                w - ширина изображения;
                c - количество каналов (для MNIST = 1).
            y (np.array[float]): Массив меток для каждого батча размера, закодированных
                с использованием one-hot encoding.
        """
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size))
        
        # Определение индексов элемнтов, из которых будет
        # создан батч
        start_idx = index * self.batch_size * self.dim[0]
        finish_idx = (index + 1) * self.batch_size * self.dim[0]
        
        # Сохранение части DataFrame для формирования батча
        tmp_df = self.df[start_idx:finish_idx]
        
        for i in range(self.batch_size):
            for img_num in range(self.dim[0]):
                # Получения одного изображения из DataFrame
                img = tmp_df.iloc[i*self.dim[0]+img_num].drop('label')
                
                # Преобразование изображения в NumPy array, преобразование
                # размера в (width, height)
                img = np.array(img).reshape(self.dim[1], self.dim[2], self.dim[3])
                
                # Сохранение изображения в батче
                X[i, img_num] = img
                
            # Получение лейблов первых двух изображений и их суммирование
            y[i] = sum(tmp_df[i*self.dim[0]:i*self.dim[0]+2]['label'])
            
        return X, keras.utils.to_categorical(y, num_classes=19)


class DataGeneratorRegression(keras.utils.Sequence):
    """
    Генератор данных для изображений из MNIST.
    Формирует батчи в виде ряда следующей размерности: 
    (размер батча, количество изображений, высота, ширина).
    Метка - сумма первых двух чисел ряда.
    В качестве метки возвращается число типа np.float32.
    """
    def __init__(self, data_frame, batch_size=32, dim=(10, 28, 28, 1)):
        """
        Конструктор класса DataGenerator.
        Parameters:
            data_frame (pd.DataFrame): DataFrame, из которого формируются батчи;
            batch_size (int): Размер батча;
            dim (tuple): Размерность вида (Количество изображений, высота, ширина).
        """
        self.df = data_frame
        self.batch_size = batch_size
        self.dim = dim
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """ Перемешивает датасет """
        self.df = self.df.sample(frac=1)        
    
    def __len__(self):
        """ Возвращает количество батчей на одну эпоху """
        return int(np.floor(len(self.df)/(self.batch_size*self.dim[0])))
    
    def __getitem__(self, index):
        """ 
        Возвращает батч изображений
        Returns:
            X (np.array): Батч данных в формате (bs, seq_len, h, w, c),
                где bs - размер батча;
                seq_len - длина последовательности изображений;
                h - высота изображения;
                w - ширина изображения;
                c - количество каналов (для MNIST = 1).
            y (np.array[float]): Массив меток для каждого батча размера.
        """
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size))
        
        # Определение индексов элемнтов, из которых будет
        # создан батч
        start_idx = index * self.batch_size * self.dim[0]
        finish_idx = (index + 1) * self.batch_size * self.dim[0]
        
        # Сохранение части DataFrame для формирования батча
        tmp_df = self.df[start_idx:finish_idx]
        
        for i in range(self.batch_size):
            for img_num in range(self.dim[0]):
                # Получения одного изображения из DataFrame
                img = tmp_df.iloc[i*self.dim[0]+img_num].drop('label')
                
                # Преобразование изображения в NumPy array, преобразование
                # размера в (width, height)
                img = np.array(img).reshape(self.dim[1], self.dim[2], self.dim[3])
                
                # Сохранение изображения в батче
                X[i, img_num] = img
                
            # Получение лейблов первых двух изображений и их суммирование
            y[i] = sum(tmp_df[i*self.dim[0]:i*self.dim[0]+2]['label'])
            
        return X, y