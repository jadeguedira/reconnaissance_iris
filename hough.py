import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))

def resize_image(image, height):
    """Redimensionne l'image en gardant le ratio"""
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image, ratio

def find_iris_and_pupil(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be read.")

    img_small, ratio_small = resize_image(img, 150) 

    img_blur_small = cv2.GaussianBlur(img_small, (9, 9), 0)  # Flou gaussien pour permettre meilleure détection de contours

    # Permet d'isoler les régions plus sombres (pupille)
    img_thresh = cv2.adaptiveThreshold(img_blur_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Repère le 1er cercle
    circles_pupil = cv2.HoughCircles(img_thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                     param1=50, param2=15, minRadius=12, maxRadius=20)

    # Repère le second cercle
    edges_small = cv2.Canny(img_blur_small, 50, 150)
    circles_iris = cv2.HoughCircles(edges_small, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                    param1=30, param2=30, minRadius=60, maxRadius=80)

    img_large, ratio_large = resize_image(img, 300) #600 : A CORRIGER !!

    output_img = cv2.cvtColor(img_large, cv2.COLOR_GRAY2BGR)

    center_x = center_y = pupil_radius = iris_radius = None

    if circles_iris is not None:
        circles_iris_int = np.uint16(np.around(circles_iris))[0]  #
        if len(circles_iris_int) > 0:
            x, y, radius = circles_iris_int[0]  
            center_x = int(x * ratio_large / ratio_small)
            center_y = int(y * ratio_large / ratio_small)
            iris_radius = int(radius * ratio_large / ratio_small)
            cv2.circle(output_img, (center_x, center_y), iris_radius, (255, 0, 0), 2)  

    if circles_pupil is not None:
        circles_pupil_int = np.uint16(np.around(circles_pupil))[0]  
        if len(circles_pupil_int) > 0:
            x, y, radius = circles_pupil_int[0]  
            pupil_radius = int(radius * ratio_large / ratio_small)
            cv2.circle(output_img, (center_x, center_y), pupil_radius, (0, 255, 0), 2)  

    if center_x is not None and pupil_radius is None:
        pupil_radius = iris_radius // 4.5
        cv2.circle(output_img, (center_x, center_y), pupil_radius, (0, 255, 0), 2)  

    return output_img, (center_x, center_y, pupil_radius, iris_radius)

def get_coord_hough(nom_image):
    result_img, circle_data = find_iris_and_pupil(nom_image)
    return circle_data

if __name__ == "__main__":
    nom_image = "iris_ech/050L_3.png"
    result_img, circle_data = find_iris_and_pupil(nom_image)
    plt.imshow(result_img)
    plt.show()



