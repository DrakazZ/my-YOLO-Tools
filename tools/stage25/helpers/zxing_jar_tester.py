import os
import json
import subprocess
from pathlib import Path
from tkinter import Tk, Button, filedialog, scrolledtext
from cryptography.fernet import Fernet

# --- ZXing setup ---
ZXING_CORE_JAR = r"D:/zxing/core-3.5.3.jar"
ZXING_JAVASE_JAR = r"D:/zxing/javase-3.5.3.jar"

def decode_qr_with_zxing(image_path: str, java_path: str = "D:/Eclipse Adoptium/jdk-21.0.8.9-hotspot/bin/java.exe") -> str:
    if not all(os.path.exists(jar) for jar in [ZXING_CORE_JAR, ZXING_JAVASE_JAR]):
        return "[ZXING ERROR] Missing jar(s)."

    try:
        abs_path = Path(image_path).absolute()
        result = subprocess.run(
            [
                java_path, "-cp",
                f"{ZXING_CORE_JAR}{os.pathsep}{ZXING_JAVASE_JAR}",
                "com.google.zxing.client.j2se.Decode",
                abs_path
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # ZXing prints multiple lines; Raw result is usually at the end
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Raw result:"):
                return line.split("Raw result:")[-1].strip()
        return result.stdout.strip()
    except Exception as e:
        return f"[ZXING ERROR] {e}"

# --- Key management ---
def load_key():
    # USB key lookup can be implemented here if needed
    local_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secret.key")
    if os.path.exists(local_key_path):
        with open(local_key_path, "rb") as f:
            return f.read()
    # generate new key if none exists
    key = Fernet.generate_key()
    with open(local_key_path, "wb") as f:
        f.write(key)
    return key

# --- Full decode + decrypt ---
def process_qr_file(file_path):
    data = decode_qr_with_zxing(file_path)
    if not data or data.startswith("[ZXING ERROR]"):
        return f"QR decode failed:\n{data}"

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return f"Decoded QR is not valid JSON:\n{data}"

    key = load_key()
    if not key:
        return "Encryption key not found."

    f = Fernet(key)
    try:
        decrypted = {
            "name": f.decrypt(payload["enc_name"].encode()).decode(),
            "id": f.decrypt(payload["enc_id"].encode()).decode(),
            "class": payload.get("class", ""),
            "university": payload.get("university", ""),
        }
    except Exception as e:
        return f"Decryption failed: {str(e)}"

    return json.dumps(decrypted, indent=2)

# --- Tkinter UI for testing ---
def select_and_process():
    file_path = filedialog.askopenfilename(
        title="Select QR image",
        filetypes=(("Images/PDF", "*.png *.jpg *.jpeg *.pdf"), ("All files", "*.*"))
    )
    if not file_path:
        return
    output_text.delete(1.0, "end")
    result = process_qr_file(file_path)
    output_text.insert("end", result)

# UI setup
root = Tk()
root.title("ZXing QR Decode + Decrypt Test")
root.geometry("600x400")

btn = Button(root, text="Select QR Image/PDF", command=select_and_process)
btn.pack(pady=10)

output_text = scrolledtext.ScrolledText(root, width=70, height=20)
output_text.pack(padx=10, pady=10, fill="both", expand=True)

root.mainloop()
