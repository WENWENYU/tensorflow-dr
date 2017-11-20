# -*- coding:utf-8 -*-
"""
@author:Luo
@file:data_augmentation.py
@time:2017/11/20 16:32
"""
import cv2
import numpy as np

crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]

def random_crop(img, area_ratio, hw_disturbed):
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_disturbed, hw_disturbed)
    hw_mult = 1 + hw_delta

    w_crop = int(np.round(w * np.sqrt(area_ratio * hw_mult)))

    if w_crop > w:
        w_crop = w

    h_crop = int(np.round(h * np.sqrt(area_ratio / hw_mult)))

    if h_crop > h:
        h_crop = h

    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_image(img, x0, y0, w_crop, h_crop)

def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    rotate = cv2.getRotationMatrix2D((w/2, h/2), angle % 360, 1)
    img_rotated = cv2.warpAffine(img, rotate, (w, h))
    cv2.imshow('img_rotated', img_rotated)

    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
             angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta

        r = hw_ratio if h > w else 1 / hw_ratio

        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))

        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

def random_rotate(img, angle_range, p_crop):
    angle = np.random.uniform(-angle_range, angle_range)
    crop = False if np.random.random() > p_crop else True

    return rotate_image(img, angle, crop)

def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255

    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

def random_hsv_transform(img, hue_range, sat_range, val_range):
    hue_delta = np.random.randint(-hue_range, hue_range)
    sat_mult = 1 + np.random.uniform(-sat_range, sat_range)
    val_mult = 1 + np.random.uniform(-val_range, val_range)

    return hsv_transform(img, hue_delta, sat_mult, val_mult)

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_give):
    log_gamma = np.log(gamma_give)
    alpha = np.random.uniform(-log_gamma, log_gamma)
    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)
