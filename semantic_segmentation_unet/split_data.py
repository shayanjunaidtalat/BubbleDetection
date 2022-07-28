import cv2
import os
import random
import numpy as np
import shutil
os.chdir("..")
image_path = os.path.join(os.getcwd(),'Photos/train/input_done4/')
mask_path = os.path.join(os.getcwd(),'Photos/train/label_done4/')
image_train_path = os.path.join(os.getcwd(),'Photos/train/input_done4/training/')
image_val_path = os.path.join(os.getcwd(),'Photos/train/input_done4/validation/')
mask_train_path = os.path.join(os.getcwd(),'Photos/train/label_done4/training/')
mask_val_path = os.path.join(os.getcwd(),'Photos/train/label_done4/validation/')
#filename = str(1) + '_' + str(int(vid_capture.get(1)))
_, _, files = next(os.walk(image_path))
file_count = len(files)
# validation_index = np.random.uniform(low=0.0, high=file_count, size=int(0.2 * file_count))
# validation_index = sorted(validation_index.astype(int))
validation_index = sorted(random.sample(range(1, file_count), int(0.2 * (file_count))))
# for i in validation_index:
#     filename = str(1) + '_' + str(int(i))


for i, file_name in enumerate(os.listdir(image_path)):
    # construct full file path
    source = image_path + file_name
    if i+1 in validation_index:
        destination = image_val_path + file_name
    else:
        destination = image_train_path + file_name
    # copy only files
    if os.path.isfile(source):
        shutil.copyfile(source, destination)
        print('copied', file_name)

for file_name in os.listdir(mask_path):
    # construct full file path
    source = mask_path + file_name
    if file_name in os.listdir(image_val_path):
        destination = mask_val_path + file_name
    else:
        destination = mask_train_path + file_name
    # copy only files
    if os.path.isfile(source):
        shutil.copyfile(source, destination)
        print('copied', file_name)
ar = os.listdir(image_train_path)
ar1 = os.listdir(mask_train_path)
ar2 = os.listdir(image_val_path)
ar3 = os.listdir(mask_val_path)
diff1 = np.setdiff1d(ar1,ar)
diff2 = np.setdiff1d(ar,ar1)
diff3 = np.setdiff1d(ar2,ar3)
diff4 = np.setdiff1d(ar3,ar2)
assert len(diff1) + len(diff2) + len(diff3) + len(diff4) == 0, "Invalid Operation"
a=1