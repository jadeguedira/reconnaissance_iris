from extraction_caracteristiques import get_encodage_filtrage
import json
#from hough import get_coord_hough
from descente_de_gradient import get_coord_ddg
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))

def cree_data_positions(liste_images, methode, chemin = ''):
    data_positions = {}
    for nom_image in liste_images:
        data_positions[nom_image] = methode(chemin + nom_image)
    return data_positions

def cree_json_signatures(nom_fichier, liste_data, data_positions, chemin = ''):
    data_signatures = {}
    for nom_image in liste_data:
        nom_personne = nom_image[:-6]
        m1,m2,m3,m4, matrice_encodee = get_encodage_filtrage(chemin, nom_image, data_positions)
        data_signatures[nom_personne] = data_signatures.get(nom_personne, [])
        data_signatures[nom_personne].append(matrice_encodee.tolist())
    with open(nom_fichier,'w') as f:
        json.dump(data_signatures, f, indent=2)


if __name__ == "__main__":
    chemin_perso = 'bdd_perso/'
    liste_images_perso =os.listdir(chemin_perso)
    nom_fichier = "signatures_perso.json" #centre retenue = pupile et disque considéré = 1/2 interne de l'iris

    data_iris_perso = cree_data_positions(liste_images_perso, get_coord_ddg, chemin_perso)
    cree_json_signatures(nom_fichier, liste_images_perso, data_iris_perso, chemin_perso)

"""
Données utilisées pour nos tests de detection de caracteristiques, où les coordonnees x,y et les deux rayons sont renseignees directement :

chemin_bdd = 'iris_ech/'
liste_images_bdd = ["001L_1.png", "001L_2.png", "001L_3.png", "002L_1.png", "002L_2.png", "002L_3.png"]
data_iris_bdd={'001L_1.png': (96, 75, 16, 62), '001L_2.png': (99, 71, 18, 66), '001L_3.png': (101, 67, 18, 66),\
           '002L_1.png': (97, 74, 20, 77), '002L_2.png': (97, 73, 22, 75), '002L_3.png': (96, 72, 19, 71)} #MARCHE POUR 200x200
#Chaque iris : clé = nom_du_fichier, valeur (x,y,r1,r2) avec r1 < r2

liste_images_groupe = ["badr_1.jpg", "badr_2.jpg", "badr_3.jpg", "manon_1.jpg", "manon_3.jpg"]
liste_nos_iris = ['nos_iris/badr_1.jpg', 'nos_iris/badr_3.jpg', 'nos_iris/manon_3.jpg', 'nos_iris/manon_1.jpg']
data_nos_iris = {'nos_iris/badr_3.jpg': (132, 93, 10, 49), 'nos_iris/manon_3.jpg': (133, 86, 11, 52), 'nos_iris/manon_1.jpg': (129, 86, 12, 53), 'nos_iris/badr_1.jpg': (154, 99, 25, 103)}

"signatures_perso_pup_rp.json #centre retenue = pupile et disque considéré = 1/2 interne de l'iris
"""