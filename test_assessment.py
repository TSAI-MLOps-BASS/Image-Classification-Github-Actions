import pytest
import os
import pandas as pd

df = pd.read_csv(os.path.join(os.getcwd() + os.sep, 'metrics.csv'))
metric_dict = df.to_dict()

def test_1k_dataset_existence():
    data_dir = os.path.join(os.getcwd() + os.sep, 'data.zip')
    assert not os.path.exists(data_dir), 'dataset with 1000 images exists'

def test_2k_dataset_existence():
    data_dir = os.path.join(os.getcwd() + os.sep, 'new-labels.zip')
    assert not os.path.exists(data_dir), 'dataset with 2000 images exists'

def test_model_pth_existence():
    model = os.path.join(os.getcwd() + os.sep, 'model.pth')
    assert not os.path.exists(model), 'Model weights file of pth extension exists'

def test_model_h5_existence():
    model = os.path.join(os.getcwd() + os.sep, 'model.h5')
    assert not os.path.exists(model), 'Model weights file of h5 extension exists'

def test_train_accuracy():
    assert metric_dict['Train_accuracy'][0] > 0.7, 'Training accuracy is less than 70%'

def test_val_accuracy():
    assert metric_dict['Val_accuracy'][0] > 0.7, 'Validation accuracy is less than 70%'

def test_cat_class_accuracy():
    assert metric_dict['Cat_accuracy'][0] > 0.7, 'Accuracy for class Cat is less than 70%'

def test_dog_class_accuracy():
    assert metric_dict['Dog_accuracy'][0] > 0.7, 'Accuracy for class Dog is less than 70%'