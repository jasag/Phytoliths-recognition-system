from darkflow.net.build import TFNet
import cv2
from skimage import io

options = {"model" : "cfg/tiny-yolo-1c.cfg", "load": -1, "threshold": 0.0}

tfnet = TFNet(options)

imgcv = cv2.imread("./images/2017_5_17_17_56Image_7636.jpg")
imgski = io.imread("./images/2017_5_17_17_56Image_7636.jpg")

#imgcv = cv2.imread("./preview.png")
#imgski = cv2.imread("./preview.png")

result = tfnet.return_predict(imgcv)
result2 = tfnet.return_predict(imgski)


print("1", result)
print("2", result2)
