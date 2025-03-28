import cv2
import numpy as np

def get_countour(mask):
    largest_contour = None
    mask_img = mask['segmentation'].astype(np.uint8) * 255
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def compute_roundness(mask) -> float:
    roundness = 0
    largest_contour = get_countour(mask)
    if largest_contour is not None:
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            roundness = (4 * np.pi * area) / (perimeter ** 2)
    return roundness

def compute_eccentricity(mask) -> float:
    eccentricity = 0
    largest_contour = get_countour(mask)
    if largest_contour is not None:
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
        return eccentricity

def compute_meters(mask) -> float:
    return 1


def compute_percentage(mask) -> float:
    return 1


def compute_pixels(mask) -> float:
    return 1
