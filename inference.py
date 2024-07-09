import torch
import cv2 as cv

model = torch.hub.load('.', 'custom', 
                        path='E:\\pyproject\\yolov5-6.0\\runs\\train\\exp\\weights\\best.pt', 
                        source='local')
# BGR RGB
img = cv.imread("E:\\pyproject\\yolov5-6.0\\data\\dataset\\test\\images\\27_jpg.rf.e2ce07bfabc1b569c6e21c502322f101.jpg")
results = model(img)
results.save()
