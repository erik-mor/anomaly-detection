import os
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

train_img_list = []
validation_img_list = []
train_labels = []
validation_labels = []

data_path = 'data/'
data_dir_list = os.listdir(data_path)

for dataset in tqdm(data_dir_list, colour='white', desc='Processing...'):
    data_type = os.listdir(data_path + '/' + dataset)
    if len(data_type) > 2:
        data_type = [tp for tp in data_type if os.path.splitext(tp)[1] == '']
    for tp in data_type:
        data_class = os.listdir(data_path + '/' + dataset + '/' + tp)
        for cl in data_class:
            img_list = os.listdir(data_path + '/' + dataset + '/' + tp + '/' + cl)
            for img in img_list:
                if os.path.splitext(img)[1] == '.db':
                    continue
                input_img = cv2.imread(data_path + '/' + dataset + '/' + tp + '/' + cl + '/' + img)
                # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize = cv2.resize(input_img, (98, 98))
                input_img_resize = np.array(input_img_resize)

                if input_img_resize.shape != (98, 98, 3): print("Error")

                if tp.strip().lower() == 'train':
                    train_img_list.append(input_img_resize)
                    train_labels.append(0 if cl.strip().lower() == 'good' else 1)
                else:
                    validation_img_list.append(input_img_resize)
                    validation_labels.append(0 if cl.strip().lower() == 'good' else 1)

train_img_list = np.array(train_img_list)
train_labels = np.array(train_labels)

validation_labels = np.array(validation_labels)
validation_img_list = np.array(validation_img_list)

train_img_list = train_img_list.astype('float32') / 32
validation_img_list = validation_img_list.astype('float32') / 32

print(f"Training set shape: {train_img_list.shape}")
print(f"Validation set shape: {validation_img_list.shape}")

fail = np.count_nonzero(train_labels) + np.count_nonzero(validation_labels)
good = len(train_labels) + len(validation_labels) - fail
print(f"Number of good samples: {good}")
print(f"Number of fail samples: {fail}")

np.save(f"train_data.npy", train_img_list)
np.save(f"train_labels.npy", to_categorical(train_labels))

np.save(f"validation_data.npy", validation_img_list)
np.save(f"validation_labels.npy", to_categorical(validation_labels))
