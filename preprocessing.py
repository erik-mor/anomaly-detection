import os
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

train_img_list = []
validation_img_list = []
train_labels = []
validation_labels = []

# directory with pin images
data_path = 'archive/'
data_dir_list = os.listdir(data_path)

# loop over all pin types
for dataset in tqdm(data_dir_list, colour='white', desc='Processing...'):  # tqdm used for progress bar
    data_type = os.listdir(data_path + '/' + dataset)
    if len(data_type) > 2:
        data_type = [tp for tp in data_type if os.path.splitext(tp)[1] == '']

    # loop over train and validation
    for tp in data_type:
        data_class = os.listdir(data_path + '/' + dataset + '/' + tp)

        # loop over good and fail
        for cl in data_class:
            img_list = os.listdir(data_path + '/' + dataset + '/' + tp + '/' + cl)

            # loop over images in final directories
            for img in img_list:
                if os.path.splitext(img)[1] == '.db':
                    continue
                input_img = cv2.imread(data_path + '/' + dataset + '/' + tp + '/' + cl + '/' + img)
                # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize = cv2.resize(input_img, (98, 98))
                input_img_resize = np.array(input_img_resize)

                # insert into lists based on type and class
                if tp.strip().lower() == 'train':
                    train_img_list.append(input_img_resize)
                    train_labels.append(0 if cl.strip().lower() == 'good' else 1)
                else:
                    validation_img_list.append(input_img_resize)
                    validation_labels.append(0 if cl.strip().lower() == 'good' else 1)

    validation_img_list_np = np.array(validation_img_list)
    validation_img_list_np = validation_img_list_np.astype('float32') / 32

    if not os.path.exists(f"data/{dataset}"):
        os.makedirs(f"data/{dataset}")

    np.save(f"data/{dataset}/validation_data_{dataset}.npy", validation_img_list_np)
    np.save(f"data/{dataset}/validation_labels_{dataset}.npy", to_categorical(validation_labels))
    validation_img_list = []
    validation_labels = []

# convert to numpy array and normalize
train_img_list = np.array(train_img_list)
train_img_list = train_img_list.astype('float32') / 32
np.save(f"data/train_data.npy", train_img_list)
np.save(f"data/train_labels.npy", to_categorical(train_labels))

# print(f"Training set shape: {train_img_list.shape}")
# print(f"Validation set shape: {validation_img_list.shape}")
#
# fail = np.count_nonzero(train_labels) + np.count_nonzero(validation_labels)
# good = len(train_labels) + len(validation_labels) - fail
# print(f"Number of good samples: {good}")
# print(f"Number of fail samples: {fail}")

