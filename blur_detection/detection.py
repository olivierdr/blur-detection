#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy


def fix_image_size(image: numpy.array, expected_pixels: float = 2E6):
    ratio = numpy.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: numpy.array, threshold: int = 100, model = 'laplace'):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if model == 'laplace':
        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = numpy.var(blur_map)
        return blur_map, score, bool(score < threshold)
    
    elif model == 'sobel':
        sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1)
        #blur_map = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1)
        abs_sobel64f = numpy.absolute(sobelx64f)
        blur_map = numpy.uint8(abs_sobel64f)
        score = numpy.var(blur_map)
        return blur_map, score, bool(score < 1100)  


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)
