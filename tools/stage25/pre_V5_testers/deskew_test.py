import cv2
import numpy as np
import pytesseract

def correct_orientation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    
    # If width > height, rotate 90° clockwise
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    return img


def deskew_image(img: np.ndarray) -> None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated



def detect_rotation(img: np.ndarray) -> int:
    # pytesseract returns orientation info
    osd = pytesseract.image_to_osd(img)
    for line in osd.split("\n"):
        if "Rotate" in line:
            angle = int(line.split(":")[-1].strip())
            return angle
    return 0



def deskew(img: np.ndarray) -> None:
    img = correct_orientation(img)
    deskewed_img = deskew_image(img)

    angle = detect_rotation(deskewed_img)
    if angle != 0:
        M = cv2.getRotationMatrix2D((deskewed_img.shape[1]//2, deskewed_img.shape[0]//2), -angle, 1.0)
        deskewed_img = cv2.warpAffine(deskewed_img, M, (deskewed_img.shape[1], deskewed_img.shape[0]))
    return deskewed_img