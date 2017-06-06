import unittest
from code.notebooks.Image_Labeler.Directory_Manager import Directory_Manager
import os
from random import randint
from shutil import rmtree


class Img_Lab_Tests(unittest.TestCase):
    DIRECTORIES_NUMBER = 3
    INIT_RAND = 0
    END_RAND = 128000
    CONSTANT_PATH = './test_dir/'
    DEFAULT_DIR = 'Default/'
    directories = []
    dir_manager = None

    @classmethod
    def setUpClass(cls):
        ''' Voy a crear un conjunto de directorios con nombres
        aleatorios pero que no existan previamente para crearlos
        , modificarlos o borrarlos sin ningun problema'''

        # Generamos los nombres de los directorios

        for i in range(Img_Lab_Tests.DIRECTORIES_NUMBER):
            while (True):
                rand = randint(Img_Lab_Tests.INIT_RAND,
                               Img_Lab_Tests.END_RAND)
                if not os.path.exists(Img_Lab_Tests.CONSTANT_PATH
                                              + str(rand)):
                    directories = Img_Lab_Tests. \
                        directories.append(str(rand))
                    break

        Img_Lab_Tests.dir_manager = Directory_Manager(Img_Lab_Tests.directories,
                                                      current_dir='',
                                                      default_dir=Img_Lab_Tests.DEFAULT_DIR,
                                                      path_cte=Img_Lab_Tests.CONSTANT_PATH)

        print('> Inicializado el entorno de TEST.')

    def test_create_directory(self):
        ''' Comprobamos que los directorios se han
        creado correctamente con la inicialización
        de la clase'''
        print('>> Comprobando que los directorios han sido creados correctamente.')
        for directory in Img_Lab_Tests.directories:
            self.assertTrue(os.path.exists(Img_Lab_Tests.CONSTANT_PATH + directory))
        print('>> ¡Todos los directorios han sido creados correctamente!')

    def test_current_directory(self):
        ''' Comprobamos que cambia de directorio 
        por defecto correctamente'''
        print('>> Comprobando el cambio de directorio actual.')
        Img_Lab_Tests.dir_manager.change_dir(Img_Lab_Tests.CONSTANT_PATH +
                                             Img_Lab_Tests.directories[0])

        self.assertEqual(Img_Lab_Tests.dir_manager.get_current_dir(),
                         Img_Lab_Tests.directories[0])

        print('>> ¡Directorio actual cambiado correctamente!')
        with self.assertRaises(KeyError):
            Img_Lab_Tests.dir_manager.change_dir('Default')
        print('>> Cambiar a un directorio no registrado o inexistente falla.(Correcto)')

    def test_current_directory_path(self):
        ''' Comprobamos que la nueva ruta  del 
        directorio actual es la que deseamos'''
        print('>> Comprobando que la nueva ruta  del '
              'directorio actual es la que deseamos')
        Img_Lab_Tests.dir_manager.change_dir(Img_Lab_Tests.CONSTANT_PATH +
                                             Img_Lab_Tests.directories[1])

        self.assertEqual(Img_Lab_Tests.dir_manager.get_current_dir_path(),
                         Img_Lab_Tests.CONSTANT_PATH +
                         Img_Lab_Tests.directories[1] + '/')
        print('>> ¡Nueva ruta correcta!')

    def test_gets(self):
        ''' Comprobamos que los métodos get 
        nos devuelven los valores deseados'''
        print('>> Comprobando los métodos get')
        self.assertEqual(Img_Lab_Tests.dir_manager.get_default_dir(),
                         Img_Lab_Tests.CONSTANT_PATH +
                         Img_Lab_Tests.DEFAULT_DIR)

        self.assertEqual(Img_Lab_Tests.dir_manager.get_path_cte(),
                         Img_Lab_Tests.CONSTANT_PATH)

        dir_example = 'Fitolito tipo 1'
        self.assertEqual(Img_Lab_Tests.dir_manager.get_possible_dir(dir_example),
                         Img_Lab_Tests.CONSTANT_PATH +
                         dir_example + '/')
        print('>> ¡Comprobados correctamente!')

    @classmethod
    def tearDownClass(cls):
        ''' Eliminamos las carpetas creadas 
        para realizar las distintas pruebas'''
        # for directory_name in self.directories:
        print('> Eliminando el entorno de test.')
        rmtree(Img_Lab_Tests.CONSTANT_PATH)
        print('> El entorno de test ha sido eliminado.')
        print('> Saliendo del test...')


if __name__ == '__main__':
    unittest.main()
