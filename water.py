import tensorflow as tf
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

class MRTrainer:
    def __init__(self, colume_names, label_name):
        self.colume_names = colume_names
        self.label_name = label_name


    def plot(self, x, y1, y2):
        plt.scatter(x, y1, label='$prediction$')
        plt.scatter(x, y2, label='$true$')
        plt.legend(loc='upper left', frameon=False)
        plt.show()

    def train(self, model, norm_train_data, train_labels):
        EPOCHS = 1000
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=60)
        history = model.fit(
            norm_train_data,
            train_labels,
            epochs=EPOCHS,
            validation_split = 0.2,
            verbose=1,
            callbacks=[early_stop])
        
        return history

    def build_model(self, train_dataset):
        model = keras.Sequential([
            layers.Dense(64, activation='relu',
                input_shape=[len(train_dataset.keys())]),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'],)
        return model

    def do_statics(self, dataset):
        stats = dataset.describe()
        print(stats)
        stats.pop(self.label_name)
        return stats.transpose()

    def norm(self, dataset, statics):
        return (dataset - statics['mean']) / statics['std']

    def split(self, data):
        train_dataset = data.sample(frac=0.8, random_state=0)
        test_dataset = data.drop(train_dataset.index)
        return train_dataset, test_dataset
    
    def clean(self, dataset):
        years = dataset.pop('Year')
        return dataset, years

    def importData(self, data_path):
        raw_dataset = pd.read_csv(data_path,
            names=self.colume_names,
            sep=",",
            skipinitialspace=True)
        dataset = raw_dataset.copy()
        #self.pairplot(dataset)
        #print(dataset.tail())
        return dataset

    def pairplot(self, dataset):
        sns.pairplot(dataset[["Year", "IWC", 'GDP']], diag_kind="auto", kind='reg')
        plt.show()

    def getData(self):
        return "/Users/zhousu.zs/tf/water_consumption.csv"

    def Run(self):
        data_path = self.getData()
        data = self.importData(data_path)

        train_dataset, test_dataset = self.split(data)
        train_dataset, train_years = self.clean(train_dataset)
        test_dataset, test_years = self.clean(test_dataset)
        train_statics = self.do_statics(train_dataset)

        train_labels = train_dataset.pop(self.label_name)
        test_labels = test_dataset.pop(self.label_name)

        norm_train_data = self.norm(train_dataset, train_statics)
        norm_test_data = self.norm(test_dataset, train_statics)

        model = self.build_model(norm_train_data)
        model.summary()

        history = self.train(model, norm_train_data, train_labels)

        loss, mae, mse = model.evaluate(
            norm_test_data, test_labels, verbose=1)
        test_predictions = model.predict(norm_test_data)
        print("Year\tTrue\t\tPrediction\n")
        for i in range(test_years.size):
            print("%d\t%f\t%f\n" % (
                test_years.values[i],
                test_labels.values[i],
                test_predictions.flatten()[i]))
        #self.plot(test_years, test_predictions.flatten(), test_labels)
        return



if __name__ == '__main__':
    column_names = ['Year', 'IWC', 'PCGDP',
            'Population', 'GDP', 'UrbanizationRate',
            'TIWC', 'PCWC']
    trainer = MRTrainer(column_names, 'IWC')
    trainer.Run()

# %%
