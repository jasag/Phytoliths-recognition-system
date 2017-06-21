from code.notebooks.Phytoliths_Classifier.classifier import Phytoliths_Classifier
import pickle

from scipy.stats import randint as sp_randint
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import svm#, linear_model

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    sigmoid_kernel, laplacian_kernel, chi2_kernel
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neighbors import KNeighborsClassifier

#########################################################################
# DEFINICIÓN DE VARIABLES
#########################################################################
PATH = '../../../rsc/obj/'
EXT = '.sav'
NUM_CENTERS = [10, 25, 50, 100, 150, 200] # Número de centros del cluster


PHYTOLITHS_TYPES = ['Rondel', 'Bulliform', 'Bilobate', 'Trichomas',
                    'Saddle', 'Spherical', 'Cyperaceae']

# Sustituir la anterior lista por esta si se quiere entrenar con todos
# los tipos de fitolitos y el fondo
#PHYTOLITHS_TYPES = ['Rondel', 'Bulliform', 'Bilobate', 'Trichomas',
#                    'Saddle', 'Spherical', 'Cyperaceae', 'Background']


# Especificación de parametros y distribuciones
# TODO Rellenar el resto de parámetros para todos los clasificadores
param_dist = {
    "GNB": {
        # No tiene parametros GaussianNB
    },
    "MNB": {
        "alpha": [0.1, 0.2, 0.5, 1, 1.5, 2]
    },
    "AB": {
        "base_estimator":  [DecisionTreeClassifier(max_depth=2),
                            DecisionTreeClassifier(max_depth=4),
                            DecisionTreeClassifier(max_depth=8)],
        "n_estimators": [150, 300, 600, 900],
        "learning_rate": [0.3, 0.7, 1, 1.5],
        "algorithm": ["SAMME", "SAMME.R"]
    },
    "QDA": {
        # Solo tiene un parámetro reg_param
    },
    # MLPClassifier
    "MLP": {
        "hidden_layer_sizes": [(100,), (50,50), (100, 100)],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'learning_rate': ['constant', 'invscaling'],
        'momentum': [0, 0.9],
        'nesterovs_momentum': [True, False],
        'learning_rate_init': [0.1, 0.2],
        "random_state": [0, 1, 2],
        "max_iter": [1000, 2000, 3000],
        "activation": ['identity', 'logistic', 'tanh', 'relu']
    },
    # Decission tree classifier
    "DTC": {
        "criterion": ['gini', 'entropy'],
        "splitter": ['best', 'random'],
        "max_depth": [1, 3, 5, None],
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "max_leaf_nodes": [3, 5, None],
        "random_state": [0, 1, 2]
    },
    # GaussianProcessClassifier
    "GP": {
        "max_iter_predict": [100, 500, 1000, 3000],
        "warm_start": [True, False],
        "random_state": [0, 1, 2],
        "multi_class": ['one_vs_rest', 'one_vs_one']
        # TODO Parametrizar kernels?
    },
    # Random forest
    "RF": {
        "n_estimators": [150, 300, 600, 900],
        "max_depth": [1, 3, 5, None],
        "max_features": sp_randint(1, 11),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
        "random_state": [0, 1, 2],
        "warm_start": [True, False]
    },
    # KNeighborsClassifier
    "KN": {
        "n_neighbors": sp_randint(3, 10),
        "weights": ['uniform', 'distance'],
        "algorithm": ['ball_tree', 'kd_tree', 'brute', 'auto'],
        "leaf_size": [15, 30, 50, 70]
    },
    # GradientBoostingClassifier
    "GB": {
        "n_estimators": [150, 300, 600, 900],
        "max_depth": [1, 3, 5, None],
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "subsample": [1.0, 0.8, 0.5, 0.2],
        "max_leaf_nodes": [3, 5, None],
        "warm_start": [True, False],
        "random_state": [0, 1, 2],
        "learning_rate": [0.1, 0.3, 0.7, 1, 1.5],
    },
    #SVM
    "SVM": {
        "C": [0.1, 0.2, 0.5, 1, 2, 4, 8],
        "gamma": [0.1, 1, 2, 5, 8, 10],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "random_state": [0, 1, 2],
        "degree": [3, 6, 15, 30, 50],
        "max_iter": [1000, 2000, 3000, 50000],
        "decision_function_shape": ['ovr', 'ovo']
    },
    "LSVM": {
        "C": [0.1, 0.2, 0.5, 1, 2, 4, 8],
        "loss": ['hinge', 'squared_hinge'],
        "random_state": [0, 1, 2],
        "max_iter": [1000, 2000, 3000, 5000]
    }
}
#########################################################################
# Función de utilidad
#########################################################################
# Utilidad para obtener los mejores resultados


def report(results, n_top=3):
    acc = 0
    best_acc = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            acc = results['mean_test_score'][candidate]
            if acc > best_acc:
                best_acc = acc
            print("El modelo con ranking: {0}".format(i))
            print("Puntuación media de la evaluación: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parámetros: {0}".format(results['params'][candidate]))
            print("")
    return best_acc

#########################################################################
# DEFINICIÓN DE CLASIFICADORES
#########################################################################
classifiers = {"KN": KNeighborsClassifier(),
               # linear_model.LinearRegression(),
               #  GaussianNB No tiene parametros, por lo tanto comprobar por separado.
               "GNB": GaussianNB(),
               "MNB": MultinomialNB(),
               "AB": AdaBoostClassifier(),
               "QDA": QuadraticDiscriminantAnalysis(),
               #"MLP": MLPClassifier(alpha=1e-4, verbose=10),
               "DTC": DecisionTreeClassifier(),
               "GB": GradientBoostingClassifier(),
               "RF": RandomForestClassifier(),
               "GP": GaussianProcessClassifier(kernel=1.0 * RBF(1.0)),
               "SVM": svm.SVC(probability=True),
               "LSVM": svm.LinearSVC()
               }

#########################################################################
# OBTENCIÓN DEL CONJUNTO DE DATOS
#########################################################################
phy_cls = Phytoliths_Classifier()
print("Comenzando a obtener el conjunto de datos.")
X, y = phy_cls.get_data(path='../../../rsc/img', classes= PHYTOLITHS_TYPES)
print("Finalizada la obtención de los conjuntos de datos.")
#########################################################################
# GENERACIÓN DE CLASIFICADORES
#########################################################################

print("Comenzando el entrenamiento.")
best_acc = 0
best_cls = None
# Con distintos números de centros de cluster
for n_cent in NUM_CENTERS:
    print("Obteniendo clusters y conjunto de instancias de entrenamiento.")
    print("El numero de clusters es:", n_cent)
    cluster, train_features = phy_cls.get_features_cluster(X,  n_cent)

    train_instances = phy_cls.get_training_set(cluster, X=X)
    print("Finalizada la obtencion de los clusters e instancias.")
    #####################################################################
    # RECORREMOS TODOS LOS CLASIFICADORES
    #####################################################################
    for k, clf in classifiers.items():
        print('Entrenando el  clasificador ' + k + '.')
        if len(param_dist[k]) > 0:

            # Número de iteraciones para la búsqueda de parametros
            if len(param_dist[k]) == 1:
                n_iter_search = 3
            elif len(param_dist[k]) <= 2:
                n_iter_search = 6
            elif len(param_dist[k]) <= 4:
                n_iter_search = 15
            else:
                n_iter_search = 20

            #Busqueda del clasificador con distintos parámetros
            cls = RandomizedSearchCV(clf,
                                     scoring='accuracy',
                                     cv=5,
                                     param_distributions=param_dist[k],
                                     n_iter=n_iter_search)

            cls.fit(train_instances, y)

            print("Finalizado el entrenamiento.")

            #############################################################
            # Obtención de los mejores parametros
            #############################################################
            print("Informe de los resultados:")
            acc = report(cls.cv_results_)
        else:
            cls = clf
            acc = np.mean(cross_val_score(cls, train_instances, y, cv=5))
            print("Informe de los resultados:")
            print("Precisión de ", acc)
        if acc > best_acc:
            print("Mejor hasta el momento.")
            best_acc = acc
            best_cls = cls