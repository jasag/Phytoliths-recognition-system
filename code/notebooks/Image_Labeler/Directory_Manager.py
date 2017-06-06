import os

class Directory_Manager():

    def __init__(self, directories, current_dir='',
                 default_dir = 'Default/', path_cte = '../../rsc/img/'):
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
        '''Creamos el directorio en nuestro proyecto
            si no existe'''
        if not os.path.exists(path):
            os.makedirs(path)

    def change_dir(self, path):
        # Obtenemos nombre del dir
        dir_name = path.split("/")
        dir_name = dir_name[len(dir_name)-1]
        self._current_dir = self._directories[dir_name]

    def get_current_dir_path(self):
        return self._current_dir

    def get_current_dir(self):
        # Obtenemos nombre del dir
        dir_name = self._current_dir.split("/")
        dir_name = dir_name[-2]
        return dir_name

    def get_default_dir(self):
        return self._default_dir

    def get_path_cte(self):
        return self._path_cte

    def get_possible_dir(self, dir):
        return self._path_cte + dir + '/'
