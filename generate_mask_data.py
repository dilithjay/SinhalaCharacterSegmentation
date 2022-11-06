import os
import numpy as np
import cv2
from tqdm import tqdm
import json


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_masks(image_path, json_path, save_dir):
    f = open(json_path, "r")
    data = json.load(f)

    for _, value in tqdm(data.items()):
        filename = value["filename"]
        name = filename.split(".")[0]

        image = cv2.imread(f"{image_path}/{filename}", cv2.IMREAD_GRAYSCALE)
        H, W = image.shape

        regions = value["regions"]
        if not regions:
            continue
        
        mask = np.zeros((H, W))

        # Draw the mask for each character on a single image.
        for i, region in enumerate(regions):
            x = region["shape_attributes"]["all_points_x"]
            y = region["shape_attributes"]["all_points_y"]

            # Each layer of the mask is filled with values equal to its index.
            points = np.array(list(zip(x, y)))
            cv2.fillPoly(mask, [points], color=(i + 1))
            
        np.save(f"{save_dir}/masks/{name}.npy", mask)
        cv2.imwrite(f"{save_dir}/images/{name}.png", image)

if __name__ == "__main__":
    save_dir = "data/"    
    create_dir(f"{save_dir}/images")
    create_dir(f"{save_dir}/masks")

    image_path = r'D:\DocumentAI\SinhalaCharacterSegmentation\images'
    json_path = r'D:\DocumentAI\SinhalaCharacterSegmentation\char_seg_ann.json'
    generate_masks(image_path, json_path, save_dir)
