import os

data_dir = 'ArASL_Database_54K_Final'

train_files = os.listdir(os.path.join(data_dir, 'train', 'ain'))
val_files = os.listdir(os.path.join(data_dir, 'val', 'ain'))

print(os.listdir(os.path.join(data_dir, 'train')))