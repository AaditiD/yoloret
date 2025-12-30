import os
import glob
from PIL import Image

dataset_path = '/workspaces/yoloret/IBERBIRDS_dataset/IBERBIRDS_dataset/IBERBIRDS'

def convert_yolo_to_voc(image_path, label_path, classes):
    img = Image.open(image_path)
    width, height = img.size
    with open(label_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        w = float(parts[3]) * width
        h = float(parts[4]) * height
        xmin = x_center - w / 2
        ymin = y_center - h / 2
        xmax = x_center + w / 2
        ymax = y_center + h / 2
        bboxes.append(f'{xmin} {ymin} {xmax} {ymax} {class_id}')
    return ' '.join(bboxes)

classes = ['Aegypius_monachus', 'Aquila_adalberti', 'Aquila_chrysaetos', 'Ciconia_ciconia', 'Ciconia_nigra', 'Falco_peregrinus', 'Gyps_fulvus', 'Milvus_migrans', 'Milvus_milvus', 'Neophron_percnopterus']

for split in ['train', 'val', 'test']:
    with open(f'/workspaces/yoloret/code/data_paths/iber_birds_{split}.txt', 'w') as f:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')
        for image_file in glob.glob(os.path.join(image_dir, '*.png')):
            base = os.path.basename(image_file).replace('.png', '')
            label_file = os.path.join(label_dir, base + '.txt')
            if os.path.exists(label_file):
                bboxes = convert_yolo_to_voc(image_file, label_file, classes)
                if bboxes:
                    f.write(f'{os.path.abspath(image_file)} {bboxes}\n')