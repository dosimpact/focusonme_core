import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)

chunker = ImageChunker(512, 512, 30)


input_img = cv2.imread("input/input01.png", cv2.IMREAD_COLOR)
input_img = input_img.astype(np.float32) / 255.
input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2RGB)
# cv2.imwrite('output/input_img.png',input_img)
cv2.imshow("input01",input_img)

mask = cv2.imread("input/input02.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = cv2.cvtColor(mask, cv2.IMREAD_REDUCED_GRAYSCALE_4)
# cv2.imwrite('output/mask.png',mask)
cv2.imshow("input02",mask)

input_mask = cv2.bitwise_not(mask)
# mask = (mask/256).astype('uint8')
cv2.imwrite('output/mask.png',mask)

cv2.imshow('input_mask', input_mask)
input_mask = input_mask.astype(np.float32) / 255.
input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)

cv2.waitKey();
print('processing...')
chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))
pred_imgs = model.predict([chunked_imgs, chunked_masks])
result_img = chunker.dimension_postprocess(pred_imgs, input_img)

print('completed!')
cv2.imshow('result', result_img)
result_img = cv2.convertScaleAbs(result_img, alpha=(255.0))
cv2.imwrite('output/result.png',result_img)

cv2.destroyAllWindows()
