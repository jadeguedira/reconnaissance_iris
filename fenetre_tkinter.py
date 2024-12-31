

import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time

from extraction_caracteristiques import identification, authentification, lis_json
from descente_de_gradient import get_coord_ddg, load_and_preprocess_image
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))


def inverser_slashes(chemin):
    translation_table = str.maketrans("/\\", "\\/")
    translation_table = str.maketrans('"', "'")
    return chemin.translate(translation_table)


class Fenetre (tk.Tk): 

    def __init__ (self, fichier_json, methode_segmentation, chemin_bdd = ''):
        super().__init__()
        self.geometry("800x600")
        self.configure(bg = 'black')
        self.title("Fenêtre Menu")
        self.target_size = (300, 300)

        self.data_signatures = lis_json(fichier_json)
        self.methode = methode_segmentation
        self.chemin_bdd = chemin_bdd

        self.creer_widget()

    def creer_widget(self): 

        self.texte = tk.Label(self, text = "Quel type de reconnaissance voulez vous ? ", bg="black", fg="white", font=("Bodoni MT", 18, "bold"))
        self.texte.place(x = 240, y=40 )

        self.choix = tk.StringVar()
        self.choix.set("Identification")
        self.choix1 = tk.Radiobutton(self, text= "Authentification", variable = self.choix, value = "Authentification", bg="black", fg="white", font=("Bodoni MT", 16, "bold"), highlightthickness=2, highlightbackground= "blue", command=self.update_background)
        self.choix2 = tk.Radiobutton(self, text= "Identification", variable = self.choix, value = "Identification",bg="black", fg="white", font=("Bodoni MT", 16, "bold"), highlightthickness=2, highlightbackground= "blue", command=self.update_background)
        self.choix1.place(x= 180, y = 110)
        self.choix2.place(x= 530, y = 110)

        self.text2 = tk.Label(self, text = "Entrez le lien vers votre image", bg="black", fg="white", font=("Bodoni MT", 16, "bold"))
        self.text2.place(x=290, y = 170)
        self.entree = tk.Entry(self, width=50)
        self.entree.place(x=273, y = 220)

        self.bouton_val = tk.Button(self, text = "Valider", bg="black", fg="white", font=("Bodoni MT", 16, "bold"), highlightthickness=2, highlightbackground= "blue")
        self.bouton_val.place(x = 370, y = 270)
        self.bouton_val.bind("<Button-1>", self.recherche)

        # images pour la décoration 

        self.image1 = Image.open('images_tkinter/iris_tkinter.jpg')
        self.image1 = self.image1.resize((400,200))
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.canva1 = tk.Canvas(self, width=400, height=200, highlightthickness=0)
        self.canva1.create_image(0, 0, anchor=tk.NW, image=self.photo1)
        self.canva1.place(x = 0, y = 400)
        self.canva1.image = self.photo1

        self.canva_im2 = tk.Canvas(self, width=400, height=200, highlightthickness=0)
        self.canva_im2.create_image(0, 0, anchor=tk.NW, image=self.photo1)
        self.canva_im2.place(x=400, y = 400)
        self.canva_im2.image = self.photo1

    def update_background(self):
        # Fonction qui change la couleur de fond des boutons radio en fonction du choix sélectionné pour que le choix soit plus lisible
        if self.choix.get() == "Authentification":
            self.choix1.config(bg="gray")
            self.choix2.config(bg="black")
        else:
            self.choix1.config(bg="black")
            self.choix2.config(bg="gray")

    def recherche(self, event):
        # Fonction qui récupère le lien vers l'image et ouvre le mode authentification ou bien identification
    
        self.lien_image = self.entree.get()
        self.lien_modifie = inverser_slashes(self.lien_image)

    
        self.texte.destroy()
        self.choix1.destroy()
        self.choix2.destroy()
        self.text2.destroy()
        self.bouton_val.destroy()
        self.entree.destroy()

        if self.choix.get() == "Authentification": 
            self.authentif()
        else : 
            self.identif()

        


    def authentif(self):
        # Fonction qui demande le prenom de la personne à authentifier  

        self.titre = tk.Label(self, text = "Mode Authentification", bg="black", fg="white", font=("Bodoni MT", 18, "bold"))
        self.titre.place(x=300, y=50)

        self.texte_auth = tk.Label(self, text = "Entrez le prénom de la personne \n à authentifier", bg="black", fg="white", font=("Bodoni MT", 16, "bold"))
        self.texte_auth.place(x=60, y = 100)

        self.nom1 = tk.Entry(self)
        self.nom1.insert(0, "prenom")
        self.nom1.place(x=110, y = 180)

        self.valider = tk.Button(self, text= "Valider", bg="black", fg="white", font=("Bodoni MT", 16, "bold"), highlightthickness=2, highlightbackground= "blue")
        self.valider.place(x=190, y = 250)
        self.valider.bind("<Button-1>", self.go_authentifier)

    def go_authentifier(self, event):
        # Fonction qui authentifie la personne et affiche le résultat 

        nom_entre = self.nom1.get().strip()
        est_authentifiee, self.x, self.y, self.rp, self.rg = authentification(self.lien_modifie, self.data_signatures, nom_entre, self.methode, get_coords = True)

        if est_authentifiee: 
            self.reponse = tk.Label(self, text = "   La personne est bien authentifiée    ", bg="black", fg="white", font=("Bodoni MT", 16, "bold"))
        else : 
            self.reponse = tk.Label(self, text = "La personne presentée ne correspond pas", bg="black", fg="white", font=("Bodoni MT", 16, "bold"))
        self.reponse.place(x=80, y = 320)

        self.affiche_segmentation()


    def identif(self):
        # Fonction qui identifie une personne et affiche son prénom 

        self.titre = tk.Label(self, text = "Mode Identification", bg="black", fg="white", font=("Bodoni MT", 18, "bold"))
        self.titre.place(x=250, y=50)

        nom_identifiee, self.x, self.y, self.rp, self.rg = identification(self.lien_modifie, self.data_signatures, self.methode, get_coords = True)
        print(self.x)

        self.reponse = tk.Label(self,text = f"La personne que vous recherchez est : \n {nom_identifiee}", bg="black", fg="white", font=("Bodoni MT", 16, "bold"))
        self.reponse.place(x=50,y=100)

        self.affiche_segmentation()

    def affiche_segmentation(self):
        # Fonction qui affiche l'image utilisée avec la ségmentation effectuée 

        self.image_segmentee = Image.open(self.lien_modifie)
        self.image_segmentee.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image_segmentee)
        x_new, y_new  = self.image_segmentee.size
        
        
        self.canva_image = tk.Canvas(self, width = x_new, height = y_new, highlightthickness=2, highlightbackground= "red")
        self.canva_image.create_image(0, 0, anchor=tk.NW, image=self.photo)
        x,y,rp,rg = self.x, self.y, self.rp, self.rg
        self.canva_image.create_oval(x-rp, y-rp, x+rp, y+rp, outline="red")  # premier cercle
        self.canva_image.create_oval(x-rg, y-rg, x+rg, y+rg, outline="blue")  # deuxième cercle


        self.canva_image.place(x=420, y = 100)
        self.canva_image.image = self.photo
    


if __name__ == "__main__":
    nom_fichier = "signatures_perso.json"
    chemin_perso = '' #'bdd_perso/'
    app = Fenetre(nom_fichier, get_coord_ddg, chemin_perso)
    app.mainloop()
        

