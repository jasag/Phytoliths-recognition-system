import os

class Directory():
    '''Creamos el directorio en nuestro proyecto
    si no existe'''
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            os.makedirs(self._path)