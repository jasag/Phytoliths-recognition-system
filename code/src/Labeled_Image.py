# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:21:48 2017

@author: Jaime Sagüillo Revilla
"""
import matplotlib.pyplot as plt
import numpy as np

class Labeled_Image:
    
    ''' Constructor en el que establecemos el clasificador'''
    def __init__(self, clf):
        self._clf = clf
    
    ''' Método para establecer una imagen y clasificarla'''
    def set_image(self, image):
        
        self._image = image
        self._indices, self._patches, self._patches_hog, self._labels = self._clf.labeler(image)
        self._indices = np.array(self._indices)
        
    def set_classifier(self, clf):
        self._clf = clf
        
    ''' Método para la obtención de las ventanas en función de 
    las probabilidades, sin Non-Maximun Suppresion'''    
    def boxes_generator_wrapper(self, probs):
        return self._clf.boxes_generator(self._indices, self._labels, probs)
    
    ''' Método para la obtención de las ventanas en función de 
    las probabilidades, con Non-Maximun Suppresion'''
    def boxes_generator_with_nms(self, probs):
        boxes = self._clf.boxes_generator(self._indices, self._labels, probs)
        self._boxes = self._clf.redundant_windows_deleter(boxes, alfa=0.3)
        return self._boxes
    
    ''' Método que muestra los resultados de la clasificación 
    de la imagen'''
    def plotter(self):
        # Mostramos las imagenes resultante
        fig, ax = plt.subplots()
            
        ax.imshow(self._image , cmap='gray')
        ax.axis('off')
        
        Nj = self._clf.get_Nj()
        Ni = self._clf.get_Ni()
        
        for i, j, _, _ in self._boxes:
            ax.add_patch(plt.Rectangle((i, j), Nj, Ni, edgecolor='red',
                                            alpha=0.3, lw=2, facecolor='none'))
