# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:21:48 2017

@author: Jaime Sagüillo Revilla
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale

class Labeled_Image:
    
    ''' Constructor en el que establecemos el clasificador'''
    def __init__(self, clf, alfa = 0.3, probs=0.8):
        # Parametro que detemina el umbral de solapamiento
        # para Non-Maximun Suppresion
        self._probs = probs
        self._alfa = alfa
        self._clf = clf
        self._indices, self._patches, self._patches_hog, \
        self._labels, self._original_image, self._rescaled_image = (0,0,0,0,None, None)

    # Probalidades
    def set_probs(self, probs):
        self._probs = probs

    def get_probs(self):
        return self._probs

    # Probalidades
    def set_alfa(self, alfa):
        self._alfa = alfa

    def get_alfa(self):
        return self._alfa

    # Image setters y utilidades
    def image_rescale(self, scale):
        self._rescaled_image = rescale(self._original_image, scale)

    def set_image(self, image):
        # Hacemos el set a ambas copias de la imagen
        self._original_image = image
        self._rescaled_image = image

    def get_rescaled_image(self):
        return self._rescaled_image

    def get_original_image(self):
        return self._original_image

    '''Método para clasificar la imagen'''
    def predict(self):
        self._indices, self._patches, self._patches_hog, self._labels = self._clf.labeler(self._rescaled_image)
        self._indices = np.array(self._indices)

    def set_classifier(self, clf):
        self._clf = clf

    ''' Método para la obtención de las ventanas en función de 
    las probabilidades, con Non-Maximun Suppresion'''
    def boxes_generator_with_nms(self):
        self._boxes = self._clf.boxes_generator(self._indices, self._labels, self._probs, self._alfa)
        return self._boxes
    
    ''' Método que muestra los resultados de la clasificación 
    de la imagen'''
    def plotter(self):
        # Mostramos las imagenes resultante
        fig, ax = plt.subplots()

        ax.imshow(self._rescaled_image , cmap='gray')
        ax.axis('off')
        
        Ni, Nj = self._clf.get_patch_size()

        for i, j, _, _ in self._boxes:
            ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                                            alpha=0.3, lw=2, facecolor='none'))
