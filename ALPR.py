# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:47:53 2022

@author: Abdullah MEMISOGLU

"""

import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from itertools import groupby

# filepath = PATH_OF_TESSERACT.EXE
filepath = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = filepath

########### YOLO MODEL #################
with open('yolo_utils/classes.names') as f:
    labels = [line.strip() for line in f]

# loading the cfg file and trained yolo weights
network = cv2.dnn.readNetFromDarknet('yolo_utils/darknet-yolov3.cfg',
                                     'yolo_utils/lapi.weights')
# get all the network layers
layers_names_all = network.getLayerNames()
#layers_names_all[i - 1]
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
# set probability minimum value
probability_minimum = 0.5
# set threshold value
threshold = 0.3
# select random colors for different objects
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
im_num = 5
################################################

## CHECK IF THE ALL MEMBERS OF THE LIST ARE EQUAL OR NOT METHOD
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

## CROP THE LICENCE PLATE FROM THE FRAME USING YOLO MODEL
def get_cropped_images(image_BGR_1, results, class_numbers, bounding_boxes):
    counter = 1
    if len(results) > 0:
        for i in results.flatten():
            #print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

            counter += 1

            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            crop_image = image_BGR_1[y_min:y_min + box_height, x_min:x_min + box_width]
            class_ = str(labels[int(class_numbers[i])])
            rand = str(random.randint(0, 1400000))
            name = class_ + '_' + rand + '.jpg'
            #print(name)
            #plt.imsave(name, crop_image)
            return name, crop_image

def LPR(image_BGR):

########LICENSE PLATE DETECTION ###############

    h, w = image_BGR.shape[:2]

    # Blob alma islemi neural networke optimum veriyi saglar
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)


    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    # DRAW A RECTANGLE ON THE LICENSE PLATE
    counter = 1
    if len(results) > 0:
        for i in results.flatten():
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

            # Incrementing counter
            counter += 1

            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()

            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            #cv2.putText(image_BGR, text_box_current, (x_min, y_min - 10),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    # Comparing how many objects where before non-maximum suppression
    # and left after
    #print('Total objects been detected:', len(bounding_boxes))
    rand = str(random.randint(0, 1400000))
    if counter != 1:
        name, cropped_image = get_cropped_images(image_BGR, results, class_numbers, bounding_boxes)
        #image = cv2.imread(name, 1)
        #print(image.shape[1])


        ########LICENSE PLATE RECOGNITION USING TESSERACT#####################

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        #plt.imsave('processed_image' + rand + '.jpg', gray)
        #plt.figure()
        #plt.imshow(gray, cmap='gray')
        #plt.title("Cropped License Plate")

        scale_percent = 300  # percent of original size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        area = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        nearest_img = cv2.resize(gray, dim, interpolation=cv2.INTER_NEAREST)
        bilinear = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

        temp1 = area
        temp2 = nearest_img
        temp3 = bilinear

        mask_const = 5
        blur1 = cv2.GaussianBlur(temp1, (mask_const, mask_const), 0)
        _, th1 = cv2.threshold((temp1 - (temp1 - blur1)), np.max(blur1) / 2, np.max(blur1), cv2.THRESH_BINARY)

        blur2 = cv2.GaussianBlur(temp2, (mask_const, mask_const), 0)
        _, th2 = cv2.threshold((temp2 - (temp2 - blur2)), np.max(blur2) / 2, np.max(blur2), cv2.THRESH_BINARY)

        blur3 = cv2.GaussianBlur(temp3, (mask_const, mask_const), 0)
        _, th3 = cv2.threshold((temp3 - (temp3 - blur3)), np.max(blur3) / 2, np.max(blur3), cv2.THRESH_BINARY)



        imgs = [th1, th2, th3]

        ############### OBTAIN LP AS STRING AND CLEAN THE INCORRECT CHARACTERS
        textlist = []
        for i in imgs:
            # Extracting text from image and removing irrelevant symbols from characters
            try:
                text = pytesseract.image_to_string(i, lang="eng")
                characters_to_remove = "[]?!()@—*“>+-/,'|£#%$&^_\~qwertyuıoopğüasdfghjklşizxcvbnmöç.:"
                new_string = text
                for character in characters_to_remove:
                    new_string = new_string.replace(character, "")
                #print(new_string)
            except IOError as e:
                print("Error (%s)." % e)

            textlist.append(new_string)
        print(textlist)

    detection_frame = ''
    cropped_im = ''
    LP_string = ''
    try:
        if all_equal(textlist) and textlist[0] != '':

            print(name)
            print(textlist[0])
            plt.imsave(name, cropped_image)
            plt.imsave("imageBGR.jpg", image_BGR)
            detection_frame = image_BGR
            cropped_im = name
            LP_string = textlist[0]

    except:
        detection_frame = ''
        cropped_im = ''
        LP_string = ''

    return detection_frame, cropped_im, LP_string


def main():
    filename = r'.\data\plate-detect.mp4'
    vidcap = cv2.VideoCapture(filename)


    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        count += 1
        if not success:
            break
        image = cv2.rotate(src = image, rotateCode = cv2.ROTATE_180)
        #plt.imsave('image%d.jpg' %count, image)


        df, ci, lps = LPR(image)
        if lps != '':
            print("lps: %s" %lps)
            print(ci)
        #print(df)
        #print(ci)
        #print(lps)


if __name__ == "__main__":
    main()
