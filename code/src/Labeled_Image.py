# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:21:48 2017

@author: heise
"""
import matplotlib.pyplot as plt
import numpy as np

class Labeled_Image:
    
#    ''' Constructor '''
#    def __init__(self, clf):
#        self._clf = clf

#    ''' Establecemos un clasificador, una imagen y la clasificamos'''
#    def __init__(self, clf, image):
#        
#        self._clf = clf
#        
#        self._image = image     
#        
#        #self.calculate_labels(self._image), 
#        self._indices, self._patches, self._patches_hog, self._labels = self._clf.labeler(image)
#        
#        self._boxes = 0
    
    ''' Establecemos un clasificador, una imagen y la clasificamos'''
    def __init__(self, clf):
        self._clf = clf
        self._image = 0
        self._boxes = 0
        self._image = 0
        self._indices = 0
        self._patches = 0
        self._patches_hog = 0 
        self._labels = 0
    
    ''' Establecemos una imagen y la clasificamos'''
    def set_image(self, image):
        
        self._image = image
        self._indices, self._patches, self._patches_hog, self._labels = self._clf.labeler(image)
        self._indices = np.array(self._indices)
        
    def set_classifier(self, clf):
        self._clf = clf
        
#    ''' Obtención de las ventanas en función de las probabilidades '''    
#    def boxes_generator_wrapper(self, probs):
#        return self._clf.boxes_generator(self._indices, self._labels, probs)
#    
    def boxes_generator_with_nms(self, probs):
        bounding_boxes = self._clf.boxes_generator(self._indices, self._labels, probs)
        self._boxes = self._clf.redundant_windows_deleter(bounding_boxes, alfa=0.3)
        return self._boxes
        
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
