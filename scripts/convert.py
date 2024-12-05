import json
import os
import shutil

label_convertion = {'watermark': 0, 'watermrk': 0}

def convert_to_yolo_format(json_data):

    image_width = json_data['imageWidth']
    image_height = json_data['imageHeight']
    shapes = json_data['shapes']
    yolo_data = []

    for shape in shapes:

        try:
            class_id = label_convertion[shape['label']]
        except Exception as e:
            class_id = -1

        if class_id == -1:
            continue

        points = shape['points']
        x_min = min(points[0][0], points[1][0])
        y_min = min(points[0][1], points[1][1])
        x_max = max(points[0][0], points[1][0])
        y_max = max(points[0][1], points[1][1])

        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_data

def save_yolo_format(yolo_data, output_path):
    with open(output_path, 'w') as file:
        for line in yolo_data:
            file.write(f"{line}\n")

def process_json_files(input_directory, output_directory):

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            json_file_path = os.path.join(input_directory, filename)
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            yolo_format_data = convert_to_yolo_format(data)

            image_path_name = os.path.join(input_directory, data['imagePath'])
            output_image_path_name = os.path.join(os.path.join(output_directory, 'images'), data['imagePath'])
            image_basename = os.path.splitext(data['imagePath'])[0]
            output_file_path = os.path.join(os.path.join(output_directory, 'labels'), image_basename + '.txt')
            save_yolo_format(yolo_format_data, output_file_path)
            shutil.copy(image_path_name, output_image_path_name)
            print(f"YOLO格式数据已保存到 {output_file_path}")


input_directory = 'mixed'
output_directory = 'watermark'
process_json_files(input_directory, output_directory)