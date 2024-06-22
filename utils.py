import os
import json
import cv2
import numpy as np

def normalize_polygons(polygons, image_shape):
    """
    Poligonlari [0, 1] araligina normalize eder.
    
    Args:
    - polygons: Liste içinde listeler şeklinde poligon koordinatlari.
    - image_shape: Görüntünün şekli (yükseklik, genişlik).

    Returns:
    - normalize edilmiş poligonlarin listesi.
    """
    h, w = image_shape[:2]  # Görüntünün yükseklik ve genişlik değerlerini al
    normalized_polygons = []
    for polygon in polygons:
        normalized_polygon = []
        for i in range(0, len(polygon), 2):  # X ve Y koordinatlarını ayrı ayrı işle
            normalized_x = polygon[i] / w
            normalized_y = polygon[i+1] / h
            normalized_polygon.extend([normalized_x, normalized_y])
        normalized_polygons.append(normalized_polygon)
    return normalized_polygons

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

def scale_polygon(coords, scale_x, scale_y):
    scaled_coords = []
    for coord in coords:
        for i in range(0, len(coord), 2):
            scaled_coords.append(coord[i] / scale_x)
            scaled_coords.append(coord[i + 1] / scale_y)
    return scaled_coords

def polygon_area(coords):
    x = coords[::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_bbox(coords):
    x = coords[::2]
    y = coords[1::2]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    return [min_x, min_y, max_x, max_y]

def upscale_image(image, scale_factor):
    """
    Upscales the input image by a given scale factor using cubic interpolation.

    :param image: Input image
    :param scale_factor: Factor by which the image should be upscaled
    :return: Upscaled image
    """
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    upscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return upscaled_image

def mask2polygon(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.0007 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.ravel().tolist()
        polygons.append(polygon)

    return polygons 

def get_area(polygons):
    area = polygon_area(polygons)
    return area

def get_bbox(polygons):
    bbox = polygon_bbox(polygons)
    return bbox
