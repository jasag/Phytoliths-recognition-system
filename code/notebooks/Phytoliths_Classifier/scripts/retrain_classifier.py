from code.notebooks.Phytoliths_Classifier.classifier import Phytoliths_Classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

PHYTOLITHS_TYPES = ['ph', 'bk']
NUM_CENTERS = 500

imgs_path = '../../../rsc/img'

path = '../../../rsc/obj/'
cls_path = path + 'cls.sav'
cluster_path = path + 'cluster.sav'

cls = pickle.load(open(cls_path, 'rb'))
cluster = pickle.load(open(cluster_path, 'rb'))

phy_cls = Phytoliths_Classifier()

print("Comenzando a obtener el conjunto de datos.")
X, y = phy_cls.get_data(path=imgs_path, classes= PHYTOLITHS_TYPES)
print("Finalizada la obtención de los conjuntos de datos.")


print("Las clases estan en el siguiente orden:")
print(phy_cls.get_classes())

print("Obteniendo clusters y conjunto de instancias de entrenamiento.")
print("El numero de clusters es:", NUM_CENTERS)
cluster, train_features = phy_cls.get_features_cluster(X, NUM_CENTERS,
                                                       cluster, pretrained= True)

train_instances = phy_cls.get_training_set(cluster, X=X)
print("Finalizada la obtencion de los clusters e instancias.")


print("Comenzando el entrenamiento.")
cls = phy_cls.get_trained_classifier(train_instances, y, cls)
print("Finalizado el entrenamiento.")
print("Guardando el clasificador.")

pickle.dump(cls, open(cls_path, 'wb'))
pickle.dump(cluster, open(cluster_path, 'wb'))