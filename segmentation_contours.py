import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
from skimage import io, filters, color, util
from skimage import exposure
from skimage import measure
import math
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))


def distance_entre_points(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calcul_distance_centre(contour, centre_contour): 
    liste_distance=[] #initalise une liste qui contiendra les distances de chaque point du contour a son centre 
    for coordonnee in contour : 
        distance=distance_entre_points(coordonnee, centre_contour)
        liste_distance.append(distance)
    return liste_distance

def get_ecart_type(liste_distances):
    moyenne_distance = np.mean(liste_distances)
    ecart_moyen = np.mean( np.abs(liste_distances - moyenne_distance) )
    return ecart_moyen / moyenne_distance

def get_centre(contours):
    centre = np.average(contours, axis=0)
    return centre

def extract_circle_from_image(image, radius, x_center, y_center, threshold=0.070) :
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

if __name__ == "__main__":
    image_name="bdd_perso/jade_1.jpg"
    eye_image = cv2.imread(image_name)  #lecture de l'image avec openCV
    eye_image_orig = eye_image          #stockage de l'image originale
    eye_image = io.imread(image_name, as_gray=True) #transformer en niveau de gris 
    l = filters.threshold_otsu(eye_image)
    eye_image= util.img_as_ubyte(eye_image > l)  #convertir en noir et blanc
    
    contours=measure.find_contours(eye_image, 0.1) #on detecte tous les contours , on stocke dans une liste, pour chaque contours une liste des coordonées de chaque points 
    fig, ax = plt.subplots()
    ax.imshow(eye_image, cmap=plt.cm.gray)

    meilleur_contour=None #initilise le contour le plus circulaire, pour l'instant il n' y en a pas 
    meilleur_ecart_type=1000  #initialise le meilleur ecart type, intialement on prend une grande valeur
    
    for indice, contour in enumerate(contours):    #on parcours chaque contours
         if len(contour) > 1000:   #élimine les contour trop petit 
            centre = get_centre(contour)  #trouve le centre du contour
            
            liste_distance=calcul_distance_centre(contour, centre) #liste avec la distance de chaque points du contour à son centre 
            ecart_type = get_ecart_type(liste_distance)
           
            if ecart_type < meilleur_ecart_type :  #si l'ecart type des distances pour ce contour est plus faible que le "meilleure" jusque là, alors ce contour est le nouveau meilleur contour 
                meilleur_ecart_type=ecart_type
                meilleur_contour=contour
                liste=liste_distance
            #ax.plot(centre[1], centre[0],'ro',linewidth=1000)
            #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)  #affiche tous les contours detectés 

    centre_iris=get_centre(meilleur_contour)   #centre_iris[0] est la coordonnée y et centre_iris[1] la coordonnée x du centre 
    distance_aucentre = calcul_distance_centre(meilleur_contour, centre_iris)  #liste des distances de chaque point du contour au centre
    rayon_iris=np.mean(distance_aucentre)  #la moyenne des rayons (distance au centre) (donc le rayon de l'iris )
    ax.plot(centre_iris[1], centre_iris[0],'ro',linewidth=5) #affichage du centre
    ax.plot(meilleur_contour[:, 1], meilleur_contour[:, 0], linewidth=2) # affichage du "cercle"
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    
