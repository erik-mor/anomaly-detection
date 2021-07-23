import os
import cv2
import numpy as np
import tensorflow as tf
import csv
import argparse


class DirExists(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentError(self, "{0} is not a valid path".format(prospective_dir))
        else:
            setattr(namespace,self.dest,prospective_dir)


class Image:
    def __init__(self, filename, img_array):
        self.filename = filename
        self.img_array = img_array


def go_deep(dirname):
    for root, subdirs, files in os.walk(dirname):

        for filename in files:
            if os.path.splitext(filename)[-1] != '.jpg':
                continue
            input_img = cv2.imread(os.path.join(root, filename))
            input_img_resize = cv2.resize(input_img, (98, 98))
            input_img_resize = [input_img_resize]
            input_img_resize = np.array(input_img_resize)
            input_img_resize = input_img_resize.astype('float32') / 255

            parse = filename.split(";")
            pin_name = parse[5]

            img = Image(filename, input_img_resize)
            if pin_name in img_dir:
                img_dir[pin_name].append(img)
            else:
                img_dir[pin_name] = [img]

        if len(subdirs):
            for subidr in subdirs:
                go_deep(os.path.join(root, subidr))


parser = argparse.ArgumentParser()
parser.add_argument("source", help="Path to the directory with image files", action=DirExists)
parser.add_argument("destination", help="Path to the directory where to save results", action=DirExists)
args = parser.parse_args()

source_dir = args.source
dest_dir = args.destination

img_dir = {}
go_deep(source_dir)

system_1 = tf.keras.models.load_model("models/vgg-3c1f-1625580569")
system_2 = tf.keras.models.load_model("models/vgg-3c1f-1625580569")

rows = [
    ["File name", "Pin", "Bad (System 1)", "Good (System 1)", "Bad (System 2)", "Good (System 2)"]
]

for pin in img_dir.keys():

    not_implemented = False
    if not os.path.exists(f"models/weights/{pin}.npy"):
        not_implemented = True
    else:
        loaded_weights = np.load(f"models/weights/{pin}.npy", allow_pickle=True)
        for weights, layer in zip(loaded_weights, system_2.layers[-4:]):
            layer.set_weights(weights)

    for image in img_dir[pin]:
        system_1_pred = system_1.predict(image.img_array)[0]
        if not not_implemented:
            system_2_pred = system_2.predict(image.img_array)[0]
        else:
            system_2_pred = ["Nan", "Nan"]

        row = [image.filename, pin, f"{system_1_pred[0]}", f"{system_1_pred[1]}", f"{system_2_pred[0]}",
               f"{system_2_pred[1]}"]
        rows.append(row)

with open(os.path.join(dest_dir, "test.csv"), "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

