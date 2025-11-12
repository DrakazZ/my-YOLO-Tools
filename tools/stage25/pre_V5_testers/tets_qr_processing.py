import cv2
import numpy as np

def qr_grid_resample(qr_img, version=10, threshold=127):
    """
    Reconstruct QR by resampling into a clean module grid.
    qr_img: cropped & deskewed QR image
    version: QR version (1=21x21, 2=25x25, ...)
    threshold: intensity threshold for binarization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)

    # Number of modules
    modules = 21 + (version - 1) * 4
    h, w = gray.shape
    module_h, module_w = h // modules, w // modules

    clean_qr = np.zeros_like(gray)

    # Go through each grid cell
    for i in range(modules):
        for j in range(modules):
            # Extract the module region
            y1, y2 = i * module_h, (i + 1) * module_h
            x1, x2 = j * module_w, (j + 1) * module_w
            cell = gray[y1:y2, x1:x2]

            # Compute mean intensity
            mean_val = np.mean(cell)

            # Fill clean_qr cell
            color = 0 if mean_val < threshold else 255
            clean_qr[y1:y2, x1:x2] = color

    return clean_qr

img = cv2.imread('C:/Users/dkz/test_qr.png')
img = qr_grid_resample(img)
cv2.imwrite("processed.png", img)