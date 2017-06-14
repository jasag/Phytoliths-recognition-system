import unittest
from code.notebooks.Phytoliths_Classifier.classifier import Phytoliths_Classifier
from sklearn.model_selection import train_test_split


class Phytoliths_Classifier_Tests(unittest.TestCase):

    def test(self):
        TRAIN_SIZE = 0.7
        NUM_CENTERS = 100

        phy_cls = Phytoliths_Classifier()

        X, y = phy_cls.get_data(path='../rsc/img')

        # Subdivisión del conjunto
        X_train, X_test, \
        y_train, y_test = train_test_split(X, y, stratify=y,
                                           train_size=TRAIN_SIZE)

        self.assertTrue(len(X) > 0)
        self.assertTrue(len(y) > 0)

        cluster, train_features = phy_cls.get_features_cluster(X_train, NUM_CENTERS)

        train_instances = phy_cls.get_training_set(cluster, X=X_train)

        cls = phy_cls.get_trained_classifier(train_instances, y_train)

        phy_cls.evaluate_classifier(cls, cluster, X_test, y_test)

if __name__ == '__main__':
    unittest.main()
