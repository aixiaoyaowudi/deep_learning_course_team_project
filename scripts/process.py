from math import sqrt, pi
import random
import cv2
import numpy as np
import json
import string
import os


def rand_real(l, r):
    return random.random()*(r-l)+l

available_fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                   cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                   cv2.FONT_ITALIC]

def process(image, text, json_data):
    scale = min(image.shape[0], image.shape[1]) / 1000

    font = random.choice(available_fonts)
    font_scale = scale * rand_real(0.5, 2) / 2
    font_thickness = max(int(scale * 2 * rand_real(0.5, 1)), 1)

    rotate_angle = rand_real(0, 360)

    print(rotate_angle)

    font_transparity = rand_real(0.1, 0.5)

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    length = int(sqrt(text_width**2 + text_height**2))+2

    H,S,V = random.randint(0,255), random.randint(255//3*1, 255//3*3), random.randint(255//2, 255)
    color = cv2.cvtColor(np.array([[[H,S,V]]], dtype = np.uint8), cv2.COLOR_HSV2RGB)[0][0]

    text_image = np.zeros((length, length, 3), np.uint8)
    cv2.putText(text_image, text, ((text_image.shape[1]-text_width)//2, (text_image.shape[0]+text_height)//2), font,
                font_scale, tuple(map(int, color)),
                font_thickness, lineType = cv2.LINE_AA)

    rotation_matrix = cv2.getRotationMatrix2D((text_image.shape[1]//2, text_image.shape[0]//2), rotate_angle, 1)

    rotated_text_img = cv2.warpAffine(text_image, rotation_matrix, (text_image.shape[1], text_image.shape[0]),
                                      flags = cv2.INTER_CUBIC)

    min_axis_0, max_axis_0, min_axis_1, max_axis_1 = length, 0, length, 0

    for i in range(length):
        for j in range(length):
            if not (rotated_text_img[i,j]<color/2).any():
                min_axis_0 = min(min_axis_0, i)
                max_axis_0 = max(max_axis_0, i)
                min_axis_1 = min(min_axis_1, j)
                max_axis_1 = max(max_axis_1, j)

    bias_0 = random.randint(0, image.shape[0] - 1 - (max_axis_0 - min_axis_0))

    bias_1 = random.randint(0, image.shape[1] - 1 - (max_axis_1 - min_axis_1))

    for i in range(min_axis_0, max_axis_0 + 1):
        for j in range(min_axis_1, max_axis_1 + 1):
            if not (rotated_text_img[i,j]<color/2).any():
                image[i - min_axis_0 + bias_0,j - min_axis_1 + bias_1] = image[i - min_axis_0 + bias_0,j - min_axis_1 + bias_1] * (1 - font_transparity) + rotated_text_img[i,j] * font_transparity

    dict_data = {"label": "watermark", "points": [[bias_1, bias_0],
                                                  [bias_1 + max_axis_1 - min_axis_1, bias_0 + max_axis_0 - min_axis_0]],
                 "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}, "mask": None}

    json_data["shapes"].append(dict_data)

letters = string.ascii_letters + string.digits + '_-'

def gen_text(len):
    return ''.join(random.choices(letters, k = len))

domains = ['net', 'com', 'cn', 'io', 'xyz']

def gen_mark():
    if random.randint(0,1):
        return gen_text(random.randint(3,4)) + '.' + gen_text(random.randint(8,15)) + '.' + random.choice(domains)
    else:
        return gen_text(random.randint(6,12)) + '@' + gen_text(random.randint(3,7)) + '.' + random.choice(domains)

watermark_num = 3

def add_watermark(input_file_path, output_file_path, output_file_name):
    output_file_path = os.path.join(output_file_path, output_file_name)
    image = cv2.imread(input_file_path)
    filename = "test"
    json_data = {"version": "5.5.0", "flags": {}, "shapes": [],
                "imagePath": output_file_name + ".jpg",
                "imageData": None, "imageHeight": image.shape[0], "imageWidth": image.shape[1]}
    for i in range(watermark_num):
        process(image, gen_mark(), json_data)

    cv2.imwrite(output_file_path + '.jpg', image)
    with open(output_file_path + '.json', 'w') as f:
        json.dump(json_data, f)

if __name__ == '__main__':

    input_directory = 'input'
    output_directory = 'output'

    file_prefix = "gen"

    count = 0

    for filename in os.listdir(input_directory):

        count += 1

        add_watermark(os.path.join(input_directory, filename), output_directory, file_prefix + str(count))