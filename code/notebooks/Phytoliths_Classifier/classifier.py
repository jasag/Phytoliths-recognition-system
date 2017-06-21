"""
@author: Jaime Sagüillo Revilla <jaime.saguillo@gmail.com>
"""
from .ImageDataset import ImageDataset
import numpy as np
from skimage.feature import daisy
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import accuracy_score
from sklearn import svm
import warnings

class Phytoliths_Classifier:
    """Clase encargada de todas las operaciones 
    necesarias para crear un clasificador utilizando 
    Bag of Words"""
    PROGRESS_NOTIFICATION = 10
    NUM_CENTERS = 100
    PHYTOLITHS_TYPES = ['Rondel', 'Bulliform', 'Bilobate', 'Trichomas',
                        'Saddle', 'Spherical', 'Cyperaceae']

    def get_data(self,path = '../../rsc/img',
                 classes = PHYTOLITHS_TYPES):
        """Método responsable de obtener el
        conjunto de imágenes en escala de grises,
        junto a sus clases"""
        dataset = ImageDataset(path, classes)
        self._classes = dataset.getClasses()
        X, y = dataset.getData()
        X = list(map(rgb2gray, X))
        return X,y

    def get_classes(self):
        return self._classes

    def features_extractor(self, img, descriptor = daisy):
        """Método responsable de extraer las 
        características de una imagen.
        
        :param img: imagen
        :param descriptor: descriptor de características 
        :return: caracterísiticas
        """
        features = descriptor(img)
        numFils, numCols, sizeDesc = features.shape

        features = features.reshape((numFils*numCols,sizeDesc))
        return features

    def whole_features_extractor(self, X_train, descriptor = daisy):
        """Método encargado de extraer las 
        características de un conjunto de 
        imágenes
        
        :param X_train: conjunto de imágenes
        :return: características
        """
        train_features = []
        i = 0

        for img in X_train:
            if i % Phytoliths_Classifier.PROGRESS_NOTIFICATION == 0:
                print("Procesada imagen" + str(i) + "/" +
                      str(len(X_train)), end="\r")

            train_features.append(self.features_extractor(img,
                                                          descriptor))
            i += 1
        print("Procesadas todas las imágenes", end="\r")

        all_features = np.concatenate(train_features)

        return train_features, all_features

    def get_features_cluster(self, X_train, num_centers,
                             cluster= KMeans, descriptor = daisy):
        """Método responsable de obtener el 
        cluster y las caracterísiticas dado 
        un conjunto de imágenes y número 
        de centros para los clusters.
        
        :param X_train: conjunto de imágenes
        :param num_centers: número de centros 
        para el cluster
        :return: cluster y conjunto de entrenamiento 
        (características)
        """
        train_features, all_features = self.whole_features_extractor(X_train,
                                                                     descriptor)

        # Se inicializa el algoritmo de Kmeans
        # indicando el número de clusters
        warnings.filterwarnings("ignore")
        cluster = cluster(num_centers)

        # Se construye el cluster con todas las
        # características del conjunto de entramiento
        return cluster.fit(all_features), train_features

    def bow_histogram_extractor(self, imgFeatures, cluster):
        """Método responsable de obtener el histograma
        de Bag of Words dadas unas caracterísitcas de una imágen,
        un cluster y el número de centros.
        
        :param imgFeatures: caracterísitcas de una imágen
        :param cluster: cluster
        :param num_centers: número de centros
        :return: histograma de Bag of Words
        """

        num_centers = len(cluster.cluster_centers_)
        # extrae pertenencias a cluster
        pertenencias = cluster.predict(imgFeatures)
        # extrae histograma
        bow_representation, _ = np.histogram(pertenencias,
                                             bins=num_centers,
                                             range=(0, num_centers - 1))
        return bow_representation


    def get_training_set(self, cluster, X=None,
                         t_features=None, descriptor = daisy):
        """Método responsable de obtener
        el conjunto de características dado 
        un cluster y un conjunto de imágenes o
        características.
        
        Si le pasamos  un conjunto de imagenes X 
        saca las caracteristicas. Si le pasamos las 
        features usa directamente las features.
        
        :param cluster: cluster
        :param X: conjunto de imágenes
        :param t_features: conjunto de características
        :return: conjunto de características 
        """
        if t_features is None and X is None:
            raise ValueError('Se debe proveer al método al menos '
                             'del parametro X o t_features.')

        train_features = t_features

        if t_features is None:
            train_features, _ = self.whole_features_extractor(X,
                                                              descriptor)

        trainInstances = []

        for imgFeatures in train_features:
            # añade al conjunto de entrenamiento final
            trainInstances.append(self.bow_histogram_extractor(imgFeatures,
                                                               cluster))

        trainInstances = np.array(trainInstances)
        return trainInstances

    def get_trained_classifier(self, trainInstances, y_train,
                               classifier=svm.SVC(kernel='linear',
                                                  C=0.01,
                                                  probability=True)):
        """Método encargado de obtener un
        clasificador entrenado dado el
         conjunto de características, las
         predicciones y opcionalmente el 
         clasificador.
        
        :param trainInstances: conjunto de características
        :param y_train: predicciones
        :param classifier: clasificador
        :return: clasificador
        """
        return classifier.fit(trainInstances, y_train)

    def predict_image(self, cls, cluster, imgTest, descriptor = daisy):
        """Método responsable de obtener un
        vector de predicciones para una imagen.
        Dado un clasificador, el número de 
        centros del cluster y la imagen.
        
        :param cls: clasificador
        :param cluster: cluster
        :param imgTest: imagen 
        :return: vector de predicciones 
        correspondiente a cada clase
        """
        imgFeatures = self.features_extractor(imgTest, descriptor)

        testInstances = np.array(self.bow_histogram_extractor(imgFeatures,
                                                         cluster))

        return cls.predict_proba(testInstances)

    def predict_image_class(self, cls, cluster,
                            imgTest,
                            classes = PHYTOLITHS_TYPES,
                            descriptor=daisy):
        """Método responsable de obtener la 
        clase para una imagen dado un 
        clasificador, el número de centros 
        del cluster y la imagen.
        
        :param imgTest: imagen
        :param types: clases, es decir, tipos de fitolitos
        :return: clase de la imagen
        """
        return classes[np.argmax(self.predict_image(cls, cluster,
                                                    imgTest,
                                                    descriptor)[0])]

    def evaluate_classifier(self, cls, cluster, X, y_true, classes = PHYTOLITHS_TYPES):
        """Método responsable de evaluar
        la precisión de un clasificador.
        
        :param cluster: cluster
        :param X: conjunto de imágenes
        :param y_true: clases correspondientes 
        a las imágenes
        :return: precisión
        """
        num_centers = len(cluster.cluster_centers_)

        instances = self.get_training_set(cluster, X)
        y_pred = list(map(lambda X: self.predict_image_class(cls,
                                                             cluster,
                                                             X, classes), X))
        accur = accuracy_score(y_true, y_pred)
        print("Precisión %.2f" % accur)
        return accur