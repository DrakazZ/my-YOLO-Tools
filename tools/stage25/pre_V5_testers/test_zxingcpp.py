import cv2
import zxingcpp

# Load image as grayscale
img = cv2.imread("W:/stage25/debug_qr_crop.png", cv2.IMREAD_GRAYSCALE)

# Decode using zxing-cpp
results = zxingcpp.read_barcode(img)

if results:
    print("Decoded:", results.text)
else:
    print("No QR found")

