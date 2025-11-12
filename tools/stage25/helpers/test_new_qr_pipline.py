import tempfile, subprocess, os, pathlib
ZXING_CORE = r"D:/zxing/core-3.5.3.jar"
ZXING_JAVASE = r"D:/zxing/javase-3.5.3.jar"
JAVA_EXE = r"C:/Program Files/Eclipse Adoptium/jdk-21.0.8.9-hotspot/bin/java.exe"
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image




def decode_pipeline(page_image):
    # 1) Crop ROI (fast)
    crop = crop_top_right_roi(page_image)  # heuristic or YOLO bbox
    
    # 2) Try cheap decode on raw crop
    result = try_decode_with_pyzbar(crop)
    if result: return result
    
    # 3) Deskew + normalize and try again
    deskewed = deskew(crop)
    for prep in [basic_enhance, clahe_then_thresh, median_then_thresh]:
        candidate = prep(deskewed)
        # try multiple sizes
        for scale in [1.0, 2.0, 3.0]:
            scaled = cv2.resize(candidate, None, fx=scale, fy=scale, interpolation=...)
            res = try_decode_with_pyzbar(scaled)
            if res: return res
        
    # 4) Try ZXing fallback (same multi-prep approach)
    for candidate in candidates_generated_above:
        res = try_decode_with_zxing(candidate_file_path)
        if res: return res
    
    # 5) Try small rotation sweeps on best candidate
    for ang in [-10,-5,0,5,10]:
        rotated = rotate(candidate, ang)
        res = try_decode_with_pyzbar(rotated) or try_decode_with_zxing(rotated)
        if res: return res

    # 6) All failed: log for manual review
    save_failure(candidate)
    return None



def deskew_crop(gray):
    # assume gray is cropped area
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = gray.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_and_thresh(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    th = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th



def try_decode_with_pyzbar(cv_img):
    pil = Image.fromarray(cv_img) if len(cv_img.shape)==2 else Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    decoded = decode(pil)
    if decoded:
        return decoded[0].data.decode('utf-8')
    return None



def try_decode_with_zxing_cvimg(cv_img):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, cv_img)
    try:
        proc = subprocess.run([JAVA_EXE, "-cp", f"{ZXING_CORE}{os.pathsep}{ZXING_JAVASE}",
                               "com.google.zxing.client.j2se.Decode", path],
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        # parse raw result if present
        for line in out.splitlines():
            if "Raw result:" in line:
                return line.split("Raw result:")[-1].strip()
        return out.strip() or None
    finally:
        try: os.remove(path)
        except: pass
