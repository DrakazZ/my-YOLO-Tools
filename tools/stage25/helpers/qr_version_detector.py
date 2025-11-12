import cv2

def detect_qr_version(qr_img):
    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Estimate number of modules (assume nearly square)
    # Finder pattern + alignment always ensures black border alignment
    # so we can count transitions in one row/col
    row = gray[h//2, :]
    # Threshold
    _, bw = cv2.threshold(row, 127, 255, cv2.THRESH_BINARY)
    # Count black-white transitions
    transitions = sum(bw[i] != bw[i+1] for i in range(len(bw)-1))
    # Estimate module count (rough)
    modules = transitions // 2  
    # Convert to version
    version = (modules - 21) // 4 + 1
    return modules, version

img = cv2.imread('C:/Users/dkz/qrdb/qr_gAAAAABoxuwN5cFpQ1ifu5Gx4VMKHykpxt_A-QZervz5MGM5Aj6d2hBQnJR-WPRfqysxl7DD9ZDal9jKxZV5FPhHT5Wda5mxyg==.png')
modules, version = detect_qr_version(img)
print (f"Detected modules: {modules}, version: {version}")