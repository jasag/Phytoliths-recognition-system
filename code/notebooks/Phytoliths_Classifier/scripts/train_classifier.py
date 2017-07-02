from code.notebooks.Phytoliths_Classifier.classifier import Phytoliths_Classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

PHYTOLITHS_TYPES = ['ph', 'bk']
NUM_CENTERS = 500

imgs_path = '../../../rsc/img'
cls_path = '../../../rsc/obj/'

estimator = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_split=1e-07, min_samples_leaf=1,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   presort=False, random_state=None, splitter='best')
clf = AdaBoostClassifier(learning_rate=1.5, base_estimator= estimator,
                         algorithm='SAMME.R', n_estimators=600)
phy_cls = Phytoliths_Classifier()
print("Comenzando a obtener el conjunto de datos.")
X, y = phy_cls.get_data(path=imgs_path, classes= PHYTOLITHS_TYPES)
print("Finalizada la obtención de los conjuntos de datos.")

print("Las clases estan en el siguiente orden:")
print(phy_cls.get_classes())

print("Obteniendo clusters y conjunto de instancias de entrenamiento.")
print("El numero de clusters es:", NUM_CENTERS)
cluster, train_features = phy_cls.get_features_cluster(X, NUM_CENTERS)

train_instances = phy_cls.get_training_set(cluster, X=X)
print("Finalizada la obtencion de los clusters e instancias.")


print("Comenzando el entrenamiento.")
cls = phy_cls.get_trained_classifier(train_instances, y, clf)
print("Finalizado el entrenamiento.")
print("Guardando el clasificador.")

pickle.dump(cls, open(cls_path + "cls.sav", 'wb'))
pickle.dump(cluster, open(cls_path + "cluster.sav", 'wb'))