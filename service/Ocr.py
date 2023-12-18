import cv2
from paddleocr import PaddleOCR
from rapidocr_onnxruntime import RapidOCR

from service.ImageUtil import orientation


class Ocr:
    def detect_ppocr(self, image):
        img_w = 1024 if image.shape[1] > 1024 else image.shape[1]
        image = cv2.resize(image, (img_w, int(img_w * image.shape[0] / image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        angle, image = orientation(image)
        B_channel, G_channel, R_channel = cv2.split(image)
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        result = ocr.ocr(R_channel, cls=True)
        if result is not None:
            results = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    item = {"position": line[0], "text": line[1][0], "score": line[1][1]}
                    print(line[1][0])
                    results.append(item)
        return results

    def detect_ort(self, image):
        img_w = 1024 if image.shape[1] > 1024 else image.shape[1]
        image = cv2.resize(image, (img_w, int(img_w * image.shape[0] / image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        img_w = 1024 if image.shape[1] > 1024 else image.shape[1]
        image = cv2.resize(image, (img_w, int(img_w * image.shape[0] / image.shape[1])),
                           interpolation=cv2.INTER_AREA)
        angle, image = orientation(image)
        B_channel, G_channel, R_channel = cv2.split(image)
        rapid_ocr = RapidOCR()
        res, elapse = rapid_ocr(R_channel)
        if res is not None:
            results = []
            for i in range(len(res)):
                item = {"position": res[i][0], "text": res[i][1], "score": res[i][2]}
                results.append(item)
        return results
