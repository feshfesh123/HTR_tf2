import os
from net_config import ArchitectureConfig, FilePaths
import cv2
import numpy as np
from shutil import copyfile

def remove_invalid_data(start = 0, end = 1000):
    print("Data Error...")
    root = FilePaths.fnDataset
    file_list = os.listdir(root)
    chars = ArchitectureConfig.CHARS
    count = 0
    for file_name in file_list:
        if file_name.endswith(".txt"):
            label_name = os.path.join(root, file_name)
            file_image = file_name.replace("txt", "jpg")
            image_name = os.path.join(root, file_image)
            with open(label_name, encoding="utf-8-sig") as f:
                lines = f.readlines()
                word = lines[0]
                for ch in list(word):
                    if (chars.count(ch) == 0):
                        os.remove(label_name)
                        os.remove(image_name)
                        count+=1
                        break
    print("Removed ", count," wrong datas !!!")

BINARY_THREHOLD = 180

def process_image_for_ocr(file_path, des_file_path):
    scale_img(file_path, des_file_path)
    im_new = remove_noise_and_smooth(des_file_path)
    return im_new

def scale_img(file_path, des_file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # (h, w)
    (wt, ht) = ArchitectureConfig.IMG_SIZE
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    resized_image = cv2.resize(img, newSize)  # (h, w)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = resized_image
    cv2.imwrite(des_file_path, target)

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    cv2.imwrite(file_name, or_image)
    return or_image

def preprocess_all_data():
    i = 1
    for root in FilePaths.fnDataCollection:
        file_list = os.listdir(root)
        for file_name in file_list:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                # Tao duong dan
                file_path = os.path.join(root, file_name)
                des_file_path = os.path.join(FilePaths.fnDataPreProcessed, str(i) + '.jpg')
                # Xu ly image
                process_image_for_ocr(file_path, des_file_path)
                # Sao chep label
                label_path = file_path.replace(".png", ".txt")
                label_path = label_path.replace(".jpg", ".txt")
                copyfile(label_path, des_file_path.replace(".jpg", ".txt"))
                i+=1
    print("Preprocessed ", i-1, " images !")
