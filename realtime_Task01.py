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
# model.summary()

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30)

cv2.imwrite('Sketcher_img_masked.png',img_masked)
cv2.imwrite('Sketcher_mask.png',mask)

while True:
  key = cv2.waitKey()

  if key == ord('q'): # quit
    break
  if key == ord('r'): # reset
    print('reset')
    img_masked[:] = img
    mask[:] = 0
    sketcher.show()
  if key == 32: # hit spacebar to run inpainting
    cv2.imwrite('input_img_masked.png',img_masked)
    cv2.imwrite('input_mask.png',mask)

    input_img = cv2.imread("input/input_img_masked.png", cv2.IMREAD_COLOR) 
    mask = cv2.imread("input/input_mask.png", cv2.IMREAD_COLOR)


    print('mask shap from ',len(mask.shape))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print('mask shap to ',len(mask.shape))

    input_img = img_masked.copy() ## 
    input_img = input_img.astype(np.float32) / 255.

    input_mask = cv2.bitwise_not(mask) ##
    input_mask = input_mask.astype(np.float32) / 255.
    input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)
    # cv2.imshow('input_img', input_img)
    # cv2.imshow('input_mask', input_mask)
    print('processing...')
    chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

    # for i, im in enumerate(chunked_imgs):
    #   cv2.imshow('im %s' % i, im)
    #   cv2.imshow('mk %s' % i, chunked_masks[i])

    pred_imgs = model.predict([chunked_imgs, chunked_masks])
    result_img = chunker.dimension_postprocess(pred_imgs, input_img)

    print('completed!')

    cv2.imshow('result', result_img)
    result_img = cv2.convertScaleAbs(result_img, alpha=(255.0))
    cv2.imwrite('result.png',result_img)

cv2.destroyAllWindows()
