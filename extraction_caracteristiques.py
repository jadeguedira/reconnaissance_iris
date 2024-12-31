import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from random import randint
import json
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))

#Modules pour la lecture de fichier
import cv2
from PIL import Image

#Modules pour la transformee de Fourier
from scipy.fftpack import fft, ifft

#Modules pour la méthode PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from skimage import io, filters, color, util
from skimage import exposure
from skimage import measure, feature

#Importation des autres fichiers
#from hough import get_coord_hough
from descente_de_gradient import get_coord_ddg


#Fonctions de lecture de fichier/redimensionnement

def resize_matrice(matrice, hauteur, larg):

    resized_matrice = cv2.resize(matrice, (larg, hauteur), interpolation=cv2.INTER_AREA)
    return resized_matrice 

def load_and_preprocess_image(image_path, target_size=(300, 300)):
    """Loads an image, resizes it while maintaining aspect ratio, and converts it to grayscale numpy array."""
    img = Image.open(image_path).convert('L') 
    img.thumbnail(target_size, Image.Resampling.LANCZOS)  #ratio maintenue
    img = np.array(img) / 255.0  # Normalisation
    return img


#Fonctions de conversion on coordonnées polaires

def polar2cart(r, theta, center):
    """Conversion de coordonnées"""
    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):
    """Renvoie une matrice rectangulaire représentant l'iris"""
    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))
    
    return polar_img


#Fonction de filtrages / encodage

#Méthode FFT et Log-Gabor

def log_gabor_filter_1d(f, f0, sigma):
    return np.exp(-(np.log(f / f0) ** 2) / (2 * np.log(sigma / f0) ** 2))

def apply_log_gabor_filter(image, f0=18, sigma=36):
    """Applique la transformée de Fourier rapide sur l'image normalisée"""
    fft_image = fft(image, axis=1)
    
    #Crée un filtre Log-Gabor 1D
    rows, cols = image.shape
    freqs = np.fft.fftfreq(cols) * cols
    freqs_0 = np.where(freqs == 0, 1e-10, freqs)
    freqs_0 = np.abs(freqs_0)

    log_gabor = log_gabor_filter_1d(freqs_0, f0, sigma)
    
    #Application le filtre Log-Gabor
    filtered_image = fft_image * log_gabor
    
    #Application de la transformée de Fourier inverse
    filtered_image = ifft(filtered_image, axis=1) #np.real( )
    
    return filtered_image

def encode_phase(image_filtree, shape_encodage = (10,240,2)):
    """Encode l'image en prenant en compte la phase de l'image filtrée"""

    phase = np.angle(image_filtree)
    #Diviser la plage de phases en quatre quadrants
    quadrant_11 = (phase >= 0) & (phase < np.pi / 2)
    quadrant_01 = (phase >= np.pi / 2) & (phase < np.pi)
    quadrant_00 = (phase >= -np.pi) & (phase < -np.pi / 2)
    quadrant_10 = (phase >= -np.pi / 2) & (phase < 0)
    
    #Attribution des codes binaires aux quadrants
    code_binaires = np.zeros(shape_encodage)
    code_binaires[quadrant_11] = np.array([1,1])
    code_binaires[quadrant_01] = np.array([0,1])
    code_binaires[quadrant_00] = np.array([0,0])
    code_binaires[quadrant_10] = np.array([1,0])
    return code_binaires

def hamming_distance_matrix(matrix1, matrix2, nb_elems = 4800):
    """Renvoie la distance de hamming entre deux matrices encodés (filtrage)"""
    if matrix1.shape != matrix2.shape:
        raise ValueError("Les deux matrices doivent avoir la même forme.")
    
    distance_matrix = np.sum(matrix1 != matrix2) 
    return distance_matrix / nb_elems

#Méthode PCA

def cree_vecteur_signature(iris_matrice):
    """Crée le vecteur signature qui caractérise l'iris
    Explication de n_components : En spécifiant n_components dans l'initialisation de l'objet PCA,
    on demande à l'algorithme de PCA de calculer les n_components premières composantes principales."""
    # Normalisation des données
    scaler = StandardScaler()
    matrice_iris_scaled = scaler.fit_transform(iris_matrice)

    # Initialisation de l'objet PCA avec le nombre de composantes souhaitées
    n_components = 10 
    pca = PCA(n_components=n_components)

    # Application de PCA aux données normalisées
    matrice_iris_pca = pca.fit_transform(matrice_iris_scaled)

    return matrice_iris_pca

def distance_vecteurs(v1, v2):
    """Renvoie la distance euclidienne entre deux vecteurs signatures (PCA)"""
    distance_euclidienne = np.linalg.norm(v1 - v2)
    return distance_euclidienne

#Fonctions générales d'encodage à partir du nom de l'image respectivement pour le filtrage et la méthode PCA

def get_encodage_filtrage(chemin_dossier, image_name, data):
    chemin_image = chemin_dossier + image_name
    eye_image = load_and_preprocess_image(chemin_image)
    x0, y0, r1, r2 = data[image_name]
    iris_matrice = img2polar(eye_image, (x0, y0), r2, (r1+r2)//2) #remplacer (r1+r2)//2 par r1 si on veut considérer tout l'iris
    iris_matrice_normalise = resize_matrice(iris_matrice, 10, 240)
    image_filtree = apply_log_gabor_filter(iris_matrice_normalise)
    image_encodee = encode_phase(image_filtree)
    return eye_image, iris_matrice, iris_matrice_normalise, image_filtree, image_encodee

def get_encodage_PCA(chemin_dossier, image_name, data):
    chemin_image = chemin_dossier + image_name
    eye_image = load_and_preprocess_image(chemin_image)
    x0, y0, r1, r2 = data[image_name]
    iris_matrice = img2polar(eye_image, (x0, y0), r2, r1) #de même : remplacer r1 par (r1+r2)//2  si on veut considérer que la moitié interne
    iris_matrice_normalise = resize_matrice(iris_matrice, 10, 240)
    vecteur_iris = cree_vecteur_signature(iris_matrice_normalise)
    return eye_image, iris_matrice, iris_matrice_normalise, vecteur_iris 

#Fonctions d'affichage

def affichage_iris_complet(m1,m2,m3,m4,nom_image, data_iris):
    """Crée pour l'iris en parametre une fenetre affichant l'etat de l'iris au cours de toutes les etapes du traitement"""
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(m1, cmap=plt.cm.gray)
    x0, y0, r1, r2 = data_iris[nom_image]
    circle1 = plt.Circle((x0, y0), r1, color='red', fill=False, linewidth=2)
    circle2 = plt.Circle((x0, y0), r2, color='green', fill=False, linewidth=2)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.title("Ségmentation de l'iris")
    plt.subplot(2, 2, 2)
    plt.imshow(m2, cmap='gray')
    plt.title("Matrice de l'iris")

    plt.subplot(2, 2, 3)
    plt.imshow(m3, cmap='gray')
    plt.title("Matrice de l'iris normalisé")

    plt.subplot(2, 2, 4)
    plt.imshow(abs(m4), cmap='gray') #Seul le module est affiché : ce sont des complexes
    plt.title('Image après filtrage')
    plt.show()


#Execution sur tous les fichiers de liste_images pour évaluer le  modèle sur la base de données

def compare_iris_filtrage(liste_images, data_iris, chemin = ''):
    """
    La fonction affiche les distances entre toutes les combinaisons d'iris avec la methode de filtrage Log-Gabor puis encodage
    Parametres : 
        - liste_images : liste des noms des images que l'on souhaite étudier
        - data_iris : dictionnaire associant pour chaque image les 4 parametres la repérant
    """
    signatures_iris = {}
    for nom_image in liste_images:
        m1,m2,m3,m4, matrice_encodee = get_encodage_filtrage(chemin, nom_image, data_iris)
        signatures_iris[nom_image] = matrice_encodee
        #affichage_iris_complet(m1,m2,m3,m4, nom_image, data_iris) #Pour un affichage complet de l'iris aux differentes etapes
    combinaisons_iris = combinations(liste_images, 2)
    for iris_1, iris_2 in combinaisons_iris:
        s1 = signatures_iris[iris_1]
        s2 = signatures_iris[iris_2]
        print('Distance', iris_1[:-4], '-', iris_2[:-4], ':', hamming_distance_matrix(s1, s2) )

def compare_iris_PCA(liste_images, data_iris, chemin = ''):
    """
    La fonction affiche les distances entre toutes les combinaisons d'iris avec la methode PCA
    Parametres : 
        - liste_images : liste des noms des images que l'on souhaite étudier
        - data_iris : dictionnaire associant pour chaque image les 4 parametres la repérant
    """
    signatures_iris = {}
    for nom_image in liste_images:
        m1,m2,m3, vecteur_signature = get_encodage_PCA(chemin, nom_image, data_iris)
        signatures_iris[nom_image] = vecteur_signature
        #affichage_iris_complet(m1,m2,m3,vecteur_signature, nom_image, data_iris)
    combinaisons_iris = combinations(liste_images, 2)
    for iris_1, iris_2 in combinaisons_iris:
        s1 = signatures_iris[iris_1]
        s2 = signatures_iris[iris_2]
        print('Distance', iris_1[:-4], '-', iris_2[:-4], ':', distance_vecteurs(s1, s2) )


def all_distances_filtrage(liste_images, data_signatures, chemin = ''):
    """Même chose que précedemment mais avec comme parametre le dictionnaire, issu du fichier json, associant à chaque
    personne la liste des signatures des images de son iris.
    Il evalue la méthode de filtrage"""
    combinaisons_iris = combinations(liste_images, 2)
    for iris_1, iris_2 in combinaisons_iris:
        nom_personne, i_signature = iris_1[:-6], int(iris_1[-5]) - 1
        s1 = data_signatures[nom_personne][i_signature]
        nom_personne, i_signature = iris_2[:-6], int(iris_2[-5]) - 1
        s2 = data_signatures[nom_personne][i_signature]
        print('Distance', iris_1[:-4], '-', iris_2[:-4], ':', hamming_distance_matrix(s1, s2) )


#Phase d'identification et d'authentification

def get_signature(nom_image, methode, chemin = ''):
    """Renvoie la signature d'une iris avec la methode "methode" (choisie parmi Hough, Descente de gradient ou Segmentation)"""
    coord = methode(chemin + nom_image)
    x,y,rp,rg = coord
    data = {nom_image : coord}
    m1,m2,m3,m4, matrice_encodee = get_encodage_filtrage(chemin, nom_image, data)
    return matrice_encodee, x,y,rp,rg

def identification(nom_image, data_signatures, methode, chemin = '', get_coords = False):
    """Renvoie le nom de la personne identifiee dans nom_image par le modele"""
    distances_moyennes = {}
    signature_inconnu, x, y, rp, rg = get_signature(nom_image, methode, chemin)
    for nom_personne, liste_signatures in data_signatures.items():
        liste_distances_personnes = []
        for signature in liste_signatures:
            distance = hamming_distance_matrix(signature_inconnu, signature)
            if distance != 0:
                liste_distances_personnes.append( distance )
        distances_moyennes[nom_personne] = np.mean(liste_distances_personnes)
    nom_identifie = min(list(distances_moyennes.keys()), key = lambda x: distances_moyennes[x] )
    if get_coords:
         return nom_identifie, x, y, rp, rg
    return nom_identifie

def authentification(nom_image, data_signatures, nom_personne, methode, chemin = '', get_coords = False, dist_seuil = 0.43):
    """Renvoie un booléen indiquant si la personne  dans nom_image correspond à la personne nom_image"""
    signature_inconnu, x, y, rp, rg = get_signature(nom_image, methode, chemin)
    liste_distances = []
    for signature in data_signatures[nom_personne]:
        distance = hamming_distance_matrix(signature_inconnu, signature)
        if distance != 0:
            liste_distances.append( distance )
    distance_moy = np.mean(liste_distances)
    if get_coords:
         return distance_moy < dist_seuil, x, y, rp, rg
    return distance_moy < dist_seuil

def lis_json(nom_fichier):
    """Renvoie le dictionaire lu dans le fichier .json en parametre"""
    with open(nom_fichier, "r") as f:
        dico_json = json.load(f)
    for cle, valeur in dico_json.items():
        for i,v in enumerate(valeur):
            dico_json[cle][i]= np.array(v)
    '''
    for cle, valeur in dico_json.items():
        dico_json[cle]= np.array(valeur)
    '''
    return dico_json

def calcule_taux_identification(liste_images, data_signatures, methode, chemin = ''):
    """Renvoie le taux de bonne identification en evaluant sur toutes les images de liste_images"""
    nb_tests = len(liste_images)
    nb_reussis = 0
    for nom_image in liste_images:
        if identification(nom_image, data_signatures, methode, chemin) == nom_image[:-6]:
            nb_reussis += 1
    return nb_reussis / nb_tests

def calcule_taux_authentification(liste_images, data_signatures, methode, chemin = '', nb_tests_na = 3):
    """Renvoie le taux de bonne authentification en evaluant :
        - Si pour chacune des images de liste_images, la personne est correctement authentifiee
        - Si pour chaque personne, apres selection aléatoire de nb_tests_na images d'autres personnes, la personne est reconnu comme differente"""
    nb_tests = len(liste_images) + len(data_signatures.keys()) * 3
    nb_reussis = 0
    #Verification d'authentification validée
    for nom_image in liste_images:
        nom_personne = nom_image[:-6] 
        if authentification(nom_image, data_signatures, nom_personne, methode, chemin) == True:
            nb_reussis += 1
    #Verification d'authentification refusée
    for nom_personne in data_signatures.keys():
        for _ in range(nb_tests_na):
            nom_image = liste_images[randint(0, len(liste_images)-1)]
            while nom_image[:-6] == nom_personne:
                nom_image = liste_images[randint(0, len(liste_images)-1)]
            if authentification(nom_image, data_signatures, nom_personne, methode, chemin) == False:
                nb_reussis += 1
    return nb_reussis / nb_tests

#Tests finaux

if __name__ == "__main__":
    nom_fichier = "signatures_perso.json"
    chemin_perso = 'bdd_perso/'
    liste_images_perso =os.listdir(chemin_perso)
    data_signatures = lis_json(nom_fichier)

    pi = calcule_taux_identification(liste_images_perso, data_signatures, get_coord_ddg, chemin = chemin_perso)
    print("Taux d'identification : "+ str( round(pi*100,2) ) +'%' )

    pa = calcule_taux_authentification(liste_images_perso, data_signatures, get_coord_ddg, chemin = chemin_perso)
    print("Taux d'authentification : "+str( round(pa*100,2) ) +'%')

    #La ligne suivante affichage toutes les combinaisons de distances d'iris, pour plus de détails
    #all_distances_filtrage(liste_images_perso, data_signatures, chemin_perso)

    """
    Tests supplémentaires, réalisés avant instauration de l'interface graphique :


    nom_image_a_tester = "ismail_4.jpg"
    nom_identifie = identification(nom_image_a_tester, data_signatures, get_coord_ddg, chemin_perso)
    print('La personne identifiée est',nom_identifie)

    nom_personne = 'jade'
    est_authentifiee = authentification(nom_image_a_tester, data_signatures, nom_personne, get_coord_ddg, chemin_perso)
    print('La personne est',nom_personne,':',est_authentifiee)
    est_authentifiee2 = authentification(nom_image_a_tester, data_signatures, nom_personne, get_coord_ddg, chemin_perso)
    print('La personne est',nom_personne,':',est_authentifiee2)
    """

    #Tests sur la methode PCA
    """
    
    compare_iris_PCA(liste_images, data_iris, chemin1)
    """





    


#Execution du code (segmentation réalisée au préalable)

#Sur la base de données

#compare_iris_filtrage(liste_images_bdd, data_iris_bdd, chemin_bdd)
#compare_iris_PCA(liste_images, data_iris, chemin1)


#compare_iris_filtrage(liste_nos_iris, data_nos_iris)
