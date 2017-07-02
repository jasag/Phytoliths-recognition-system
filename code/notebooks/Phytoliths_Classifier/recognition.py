"""
@author: Jaime Sagüillo Revilla <jaime.saguillo@gmail.com>
"""
from .classifier import Phytoliths_Classifier

import pickle

import warnings

import numpy as np  # numpy como np

from skimage.transform import rescale
from skimage.color import rgb2gray

from skimage.feature import daisy

import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Phytoliths_Recognition:
    """Clase encargada del reconocimiento de 
    fitolitos dentro de una imagen"""
    PROGRESS_NOTIFICATION = 10
    NUM_CENTERS = 500
    PHYTOLITHS_TYPES = ['Background', 'Phytolith']
    PHYTOLITH_INDEX = 1
    path = '../../rsc/obj/'

    def __init__(self, cls_path=path+'cls.sav', cluster_path=path+'cluster.sav',
                 step_size_h=40, step_size_v=40, height=150, width=150):
        """Constructor de la clase mediante el cual cargamos
        el clasificador y cluster.
        
        :param cls_path: ruta del clasificador
        :param cluster_path: ruta del cluster
        """
        self._cls = pickle.load(open(cls_path, 'rb'))
        self._cluster = pickle.load(open(cluster_path, 'rb'))
        self._phy_cls = Phytoliths_Classifier()
        self._step_size_h = step_size_h
        self._step_size_v = step_size_v
        self._height = height
        self._width = width

    def set_step_size(self, step_size_h, step_size_v):
        self._step_size_h = step_size_h
        self._step_size_v = step_size_v

    def set_patch_size(self, height, width):
        self._height = height
        self._width = width

    def predict_window(self, image):
        """Método encargado de predecir
        las probabilidades de la pertencia a 
        una clase de un recorte de la imagen.
        
        :param image: imagen
        :return: probabilidades para cada clase
        """
        return self._phy_cls.predict_image(self._cls,
                                           self._cluster,
                                           image, daisy)[0][Phytoliths_Recognition.PHYTOLITH_INDEX]

    def sliding_window(self, image):
        """Método encargado de realizar la ventana
        deslizante a traves de toda la imagen.
        
        :return: cada uno de los recortes de la imagen
        """
        for y in range(0, image.shape[0], self._step_size_v):
            for x in range(0, image.shape[1], self._step_size_h):
                yield (x, y, image[y:y + self._height, x:x + self._width])

    def predict_sliding_window(self, image):
        """Método que recibe la imagen, la divide 
        en ventanas, las predice y devuelve las 
        ventanas y sus probabilidades.
        
        :param image: imagen a predecir
        :return: probabilidades y coordenadas de las ventanas
        """
        probs = []
        predictions = []

        i = 0
        # loop over the sliding window for each layer of the pyramid
        n_winds = 0
        for (x, y, window) in self.sliding_window(image):
            if window.shape[0] != self._height or window.shape[1] != self._width:
                continue
            n_winds += 1

        print("Ventanas a procesar ", n_winds)

        for (x, y, window) in self.sliding_window(image):
            # Ignorar ventana si no cumple con el ancho y alto definidos
            if window.shape[0] != self._height or window.shape[1] != self._width:
                continue

            i += 1
            if i % 10 == 0:
                print("Procesada" + str(i) + " ventanas de " + str(n_winds), end="\r")

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            prob = self.predict_window(window)
            if prob > 0.5:
                probs.append(prob)
                # x1 ,y1, x2, y2
                box = (x, y, x + self._width, y + self._height)
                predictions.append(box)

        return probs, np.array(predictions)

    def non_max_suppression(self, boxes, probs=None, overlapThresh=0.3):
        """Método que se encarga de la realización
        de non-maximum suppression sobre un conjunto
        de ventanas. Obtenido de <https://github.com/jrosebr1/imutils/\
        blob/master/imutils/object_detection.py>
        
        :param probs: probabilidades
        :param overlapThresh: umbral de solapamiento
        :return: coordenadas de las ventanas
        """
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        # if probabilities are provided, sort on them instead
        if probs is not None:
            idxs = probs

        # sort the indexes
        idxs = np.argsort(idxs)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes[pick].astype("int")

    def predict_sliding_window_nms(self, image, required_probs=0.995, rescale_factor=0.5, overlapThresh=0.3):
        """Método que nos permite predecir las ventanas
        que superan una determinada probabilidad y a su 
        vez se aplica la técnica non-maximum supression. 
        Obteniendo finalmente las coordenadas de las 
        ventanas.
        
        :param image: imagen
        :param required_probs: probabilidades minimas \
        de las ventanas a considerar.
        :param rescale_factor: factor de reescalado
        :param overlapThresh: umbral de solapamiento 
        :return: coordenadas de las ventanas
        """
        warnings.filterwarnings("ignore")
        image = rescale(rgb2gray(image), rescale_factor)

        probs, predictions = self.predict_sliding_window(image)

        probs = np.array(probs)
        predictions = np.array(predictions)

        predictions = predictions[probs > required_probs]
        probs = probs[probs > required_probs]
        return self.non_max_suppression(predictions, probs=probs,
                                        overlapThresh=overlapThresh)

    def plot(self, image, boxes):
        fig = plt.figure(figsize=(7,10))

        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(image, cmap=plt.get_cmap('gray'))

        for box in boxes:
            ax.add_patch(patches.Rectangle((box[0], box[1]),
                                           box[2] - box[0],
                                           box[3] - box[1],
                                           linewidth=1, edgecolor='r'
                                           , facecolor='none'))