import unittest
from code.notebooks.Phytoliths_Classifier.recognition import Phytoliths_Recognition
from skimage import io


class Phytoliths_Recognition_Tests(unittest.TestCase):

    def test(self):
        path = '../rsc/obj/'
        cls_path = path + 'cls.sav'
        cluster_path = path + 'cluster.sav'

        recog = Phytoliths_Recognition(cls_path=cls_path,cluster_path=cluster_path)
        image = io.imread('../rsc/img/Default/2017_5_17_17_54Image_746.jpg')
        boxes = recog.predict_sliding_window_nms(image, required_probs= 0.99)
        print(boxes)

if __name__ == '__main__':
    unittest.main()
