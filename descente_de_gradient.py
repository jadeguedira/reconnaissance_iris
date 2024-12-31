import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Circle
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))

def load_and_preprocess_image(image_path, target_size=(300, 300)):
    """
    Charge une image, la convertit en niveaux de gris, la redimensionne et la normalise.
    """
    img = Image.open(image_path).convert('L')
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    return img

def create_circle_image(shape, radius, x_center, y_center):
    """
    Crée une image de la taille donnée avec un cercle noir sur un fond blanc.
    """
    img = np.ones(shape)
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    img[mask] = 0
    return img

def compute_difference(image, circle_image):
    """
    Calcule la somme des différences au carré entre deux images.
    """
    return np.sum((image - circle_image)**2)

def gradient_descent(image, shape, r0, x0, y0, learning_rate=0.08, num_iterations=300):
    """
    Effectue une descente de gradient pour trouver les meilleurs paramètres d'un cercle correspondant à l'image.
    """
    min_diff = float('inf')
    best_r0, best_x0, best_y0 = r0, x0, y0
    no_improvement_counter = 0
    iteration = 0

    while iteration < num_iterations and no_improvement_counter < 100:
        circle_image = create_circle_image(shape, r0, x0, y0)
        diff = compute_difference(image, circle_image)

        if diff < min_diff:
            min_diff = diff
            best_r0, best_x0, best_y0 = r0, x0, y0
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        grad_r = (compute_difference(image, create_circle_image(shape, r0 + 1, x0, y0)) -
                  compute_difference(image, create_circle_image(shape, r0 - 1, x0, y0))) / 2.0
        grad_x = (compute_difference(image, create_circle_image(shape, r0, x0 + 1, y0)) -
                  compute_difference(image, create_circle_image(shape, r0, x0 - 1, y0))) / 2.0
        grad_y = (compute_difference(image, create_circle_image(shape, r0, x0, y0 + 1)) -
                  compute_difference(image, create_circle_image(shape, r0, x0, y0 - 1))) / 2.0

        r0 -= learning_rate * grad_r
        x0 -= learning_rate * grad_x
        y0 -= learning_rate * grad_y

        iteration += 1

    return best_r0, best_x0, best_y0, min_diff

def extract_circle_from_image(image, radius, x_center, y_center, threshold=0.070):
    """
    Extrait un cercle d'une image et corrige les taches blanches (flash) à l'intérieur, et converti en blanc tous les pixels au delà d'un certain seuil.
    """
    mask = np.zeros_like(image)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask_area = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    mask[mask_area] = 1
    extracted_circle = np.ones_like(image)
    extracted_circle[mask_area] = image[mask_area]

    extracted_circle[extracted_circle > threshold] = 1.0
    white_spots = (x - x_center)**2 + (y - y_center)**2 <= (radius * 0.2)**2
    extracted_circle[white_spots] = 0.0

    return extracted_circle

def get_coord_ddg(image_path):
    """
    Utilise la descente de gradient pour trouver les paramètres optimaux (coordonnées) d'un cercle correspondant à une image d'iris.
    """
    iris_image = load_and_preprocess_image(image_path)
    shape = iris_image.shape

    # Première descente de gradient sur l'image originale
    optimal_r, optimal_x, optimal_y, min_diff = gradient_descent(iris_image, shape, shape[0] // 4, shape[1] // 2, shape[0] // 2)

    # Extraire l'image du cercle et fixer les taches blanches
    extracted_image = extract_circle_from_image(iris_image, optimal_r, optimal_x, optimal_y)

    # Deuxième descente de gradient sur l'image extraite
    new_shape = extracted_image.shape
    new_optimal_r, new_optimal_x, new_optimal_y, new_min_diff = gradient_descent(extracted_image, new_shape, optimal_r / 3, optimal_x, optimal_y)
    """
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(m1, cmap=plt.cm.gray)
    x0, y0, r1, r2 = data_iris[nom_image]
    circle1 = plt.Circle((x0, y0), r1, color='red', fill=False, linewidth=2)
    circle2 = plt.Circle((x0, y0), r2, color='green', fill=False, linewidth=2)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    """

    return int(new_optimal_x), int(new_optimal_y), int(new_optimal_r), int(optimal_r)

if __name__ == "__main__":
    image_path = 'bdd_perso/badr_3.jpg'
    img = load_and_preprocess_image(image_path)
    x0, y0, r1, r2 = get_coord_ddg(image_path)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    circle1 = plt.Circle((x0, y0), r1, color='red', fill=False, linewidth=2)
    circle2 = plt.Circle((x0, y0), r2, color='green', fill=False, linewidth=2)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.show()
