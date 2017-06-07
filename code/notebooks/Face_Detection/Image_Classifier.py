"""
@author: Jaime Sagüillo Revilla
"""
from skimage import feature
import windows as wins
import numpy as np

class Image_Classifier:
    
    ''' Constructor'''
    def __init__(self, classifier, is_probs_classifier = False,
                 patch_size=(62, 47), istep=2, jstep=2, scale = 1.0):
        self._classifier = classifier
        self._is_probs_classifier = is_probs_classifier
        # Tamaño de la ventana en Y y X, respectivamente
        self._patch_size = patch_size
        self._istep = istep
        self._jstep = jstep
        self._scale = scale

    # Getters y setters de attribs

    # patch
    def get_patch_size(self):
        return self._patch_size

    def set_patch_size(self, patch_size):
        self._patch_size = patch_size

    #istep
    def get_istep(self):
        return self._istep

    def set_istep(self, istep):
        self._istep = istep

    #jstep
    def get_jstep(self):
        return self._jstep

    def set_jstep(self, jstep):
        self._jstep = jstep

    #classifier
    def get_classifier(self):
        return self._classifier
    
    def set_classifier(self, classifier, is_probs_classifier = False):
        self._classifier = classifier
        self._is_probs_classifier = is_probs_classifier
    
    # Métodos más complejos
    
    ''' Extractor de las ventanas de una imagen'''
    def windows_extracter(self, image):
        indices, patches = zip(*wins.sliding_window(image, patch_size = self._patch_size,
                                                    istep = self._istep, jstep = self._jstep,
                                                    scale = self._scale))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        return (indices, patches, patches_hog)
        
    '''Generador de las ventanas o cajas, con las coordenadas necesarias'''
    def boxes_generator(self, indices, labels, probs, alfa):
        boxes = list()
        Ni, Nj = self._patch_size
        for i, j in indices[labels > probs]:
            boxes.append((i, j, i+Ni, j+Nj))
        boxes = self.redundant_windows_deleter(np.array(boxes), alfa)
        return boxes

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
    