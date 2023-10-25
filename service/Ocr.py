import cv2
from rapidocr_onnxruntime import RapidOCR

from service import orientation


class Ocr:
    def __init__(self):
        self.rapid_ocr = RapidOCR()

    def detect(self, image):
        img_w = 1024 if image.shape[1] > 1024 else image.shape[1]
        image = cv2.resize(image, (img_w, int(img_w * image.shape[0] / image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        image = orientation(image)
        B_channel, G_channel, R_channel = cv2.split(image)
        res, elapse = self.rapid_ocr(R_channel)
        if res is not None:
            results = []
            for i in range(len(res)):
                item = {"position": res[i][0], "text": res[i][1], "score": res[i][2]}
                results.append(item)
        return results
