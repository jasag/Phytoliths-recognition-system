import os
import glob
from skimage import io as sio
from matplotlib import pyplot as plt
import numpy as np
from copy import copy

class ImageDataset:
    def __init__(self,path_to_dataset,restrictTo = []):
        self.path_to_dataset = path_to_dataset
        classes=os.listdir(path_to_dataset)
        self.paths = dict()

        for cl in classes: 
            current_paths=sorted(glob.glob(os.path.join(path_to_dataset,cl,"*.jpg")))
            self.paths[cl]=current_paths
            
        if len(restrictTo)>0: 
            new_paths = {cl:self.paths[cl] for cl in restrictTo}
            self.paths=new_paths
            
    def getClasses(self):
        return sorted(self.paths.keys())
    
    def getData(self):
        X = []
        Y = []
        for cl in self.getClasses(): # por cada clase
            paths=self.paths[cl]
            for path in paths:
                X.append(sio.imread(path))
                Y.append(cl)
        return  np.array(X), np.array(Y)
