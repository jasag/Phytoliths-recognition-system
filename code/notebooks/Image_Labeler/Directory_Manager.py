"""
@author: Jaime Sagüillo Revilla <jaime.saguillo@gmail.com>
"""
import os

class Directory_Manager():
    """
    Clase utilizada para gestionar los directorios en los
    que debe guardar los distintos ficheros el etiquetador
    de imágenes.
    """
    def __init__(self, directories, current_dir='',
                 default_dir = 'Default/', path_cte = '../../rsc/img/'):
        ''' Constructor de la clase que gestiona los directorios
        del etiquetador.
        
        :param directories: lista de nombres de los directorios
        :param current_dir: nombre de uno de los directorios anteriores
        que será el directorio actual para guardar los ficheros.
        :param default_dir: nombre del directorio por defecto para guardar
        las imágenes y etiquetas del etiquetador.
        :param path_cte: ruta donde se crearán los directorios pasados por 
        parametro.
        '''
        self._directories = dict()
        # Path constante
        self._path_cte = path_cte
        self._default_dir = self._path_cte + default_dir
        if current_dir != '':
            self._current_dir = self._path_cte + current_dir + '/'
        else:
            self._current_dir = self._path_cte + current_dir

        self.directory_creator(self._default_dir)

        for dir_item in directories:
            # Añadimos dir al diccionario
            self._directories[dir_item] = self._path_cte + dir_item + "/"
            # Creamos directorio
            self.directory_creator(self._directories[dir_item])


    def directory_creator(self, path):
        """ Creamos el directorio en nuestro proyecto
        si no existe.
        
        :param path: ruta completa del directorio a crear
        :return: no devuelve ningun parametro
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise ValueError('El directorio ya existe en'
                             ' la ruta especificada: ' + path)

    def change_dir(self, path):
        ''' Método que realiza el cambio 
        del directorio actual.
        
        :param path: ruta del directorio
        :return: no devuelve ningun parametro
        '''
        # Obtenemos nombre del dir
        dir_name = path.split("/")
        dir_name = dir_name[len(dir_name)-1]
        self._current_dir = self._directories[dir_name]

    def get_current_dir_path(self):
        """ Método que devuelve la ruta 
        del directorio actual.
        
        :return: ruta del directorio actual 
        """
        return self._current_dir

    def get_current_dir(self):
        """ Método que devuelve el nombre
        del directorio actual.
        
        :return:  nombre del directorio actual
        """
        # Obtenemos nombre del dir
        dir_name = self._current_dir.split("/")
        dir_name = dir_name[-2]
        return dir_name

    def get_default_dir(self):
        """ Método que devuelve la ruta del
         directorio por defecto.
        
        :return: ruta del directorio por defecto
        """
        return self._default_dir

    def get_path_cte(self):
        """ Método que devuelve la ruta sobre 
        la que cuelgan todos los directorios.
        
        :return: ruta sobre la que cuelgan todos 
        los directorios
        """
        return self._path_cte

    def get_possible_dir(self, dir):
        """ Método que devuelve la ruta de un directorio
        mediante un nombre de directorio.
        
        :param dir: nombre de un directorio
        :return: ruta del directorio
        """
        return self._path_cte + dir + '/'
