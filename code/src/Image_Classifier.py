# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:44:39 2017

@author: Jaime Sagüillo Revilla
"""
from skimage import feature
import windows as wins
import numpy as np

class Image_Classifier:
    
    ''' Constructor'''
    def __init__(self, classifier, is_probs_classifier = False, Nj = 62, Ni = 47):
        self._classifier = classifier
        self._is_probs_classifier = is_probs_classifier
        # Tamaño de la ventana en Y y X, respectivamente
        self._Nj = Nj
        self._Ni = Ni
    
    # Getters y setters de attribs
    def get_Nj(self):
        return self._Nj
    
    def set_Nj(self, Nj):
        self._Nj = Nj
        return Nj
    
    def get_Ni(self):
        return self._Ni
    
    def set_Ni(self, Ni):
        self._Ni = Ni
        return Ni
    
    def get_classifier(self):
        return self._classifier
    
    def set_classifier(self, classifier, is_probs_classifier = False):
        self._classifier = classifier
        self._is_probs_classifier = is_probs_classifier
    
    # Métodos más complejos
    
    ''' Extractor de las ventanas de una imagen'''
    def windows_extracter(self, image):
        indices, patches = zip(*wins.sliding_window(image))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        return (indices, patches, patches_hog)
        
    '''Generador de las ventanas o cajas, con las coordenadas necesarias'''
    def boxes_generator(self, indices, labels, probs):
        boxes = list()
        for i, j in indices[labels > probs]:
            boxes.append((j, i, j+self._Nj, i+self._Ni))
        return np.array(boxes)
    
    ''' Etiquetador de la imagen, es decir, método que obtiene las ventanas 
    resultantes de la clasificación de la imagen'''
    def labeler(self, image):
        indices, patches, patches_hog = self.windows_extracter(image)
        
        if self._is_probs_classifier == True:    
            labels = self._classifier.predict_proba(patches_hog)[:,1]
        else:
            labels = self._classifier.predict(patches_hog)
        
        return indices, patches, patches_hog, labels
    
    ''' Elimina mediante la técnica de Non-Maximum Supression las ventanas 
    redundantes'''
    def redundant_windows_deleter(self, bounding_boxes, alfa=0.3):
        return wins.non_max_suppression_fast(bounding_boxes, alfa)
    