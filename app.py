import os
import io
import zipfile
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from PIL import Image, ImageEnhance
from rembg import remove

app = Flask(__name__)

# -----------------------------
# Folders
# -----------------------------
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Optional nose cascade
nose_cascade_path = 'cascades/haarcascade_mcs_nose.xml'
if os.path.exists(nose_cascade_path):
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
else:
    nose_cascade = None

# -----------------------------
# Store mapping of original -> processed filename
# -----------------------------
processed_map = {}

# -----------------------------
# Image processing functions
# -----------------------------
def crop_from_nose_centered(pil_img):
    """Crop 2x2 square centered at nose (or face center as fallback)"""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_img, w_img = img.shape[:2]

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect nose if cascade available
    noses = []
    if nose_cascade is not None:
        noses = nose_cascade.detectMultiScale(gray, 1.1, 4)

    # Anchor point
    if len(noses) > 0:
        x, y, w_nose, h_nose = noses[0]
        nose_cx = x + w_nose // 2
        nose_cy = y + h_nose // 2
    elif len(faces) > 0:
        x, y, w_face, h_face = faces[0]
        nose_cx = x + w_face // 2
        nose_cy = y + h_face // 2
    else:
        nose_cx, nose_cy = w_img // 2, h_img // 2

    # Crop size
    crop_size = int(min(h_img, w_img) * 2) if len(faces) == 0 else int(max(faces[0][2], faces[0][3]) * 2)

    half_w = crop_size // 2
    half_h = crop_size // 2
    x1 = max(nose_cx - half_w, 0)
    x2 = min(nose_cx + half_w, w_img)
    y1 = max(nose_cy - half_h, 0)
    y2 = min(nose_cy + half_h, h_img)

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def remove_bg_white(pil_img):
    """Remove background and force pure white background, removing redundant pixels."""
    # Step 1: Remove background
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    removed_bytes = remove(img_bytes.getvalue())
    pil_img_rgba = Image.open(io.BytesIO(removed_bytes)).convert("RGBA")

    # Step 2: Convert to numpy array
    img_np = np.array(pil_img_rgba)

    # Step 3: Extract alpha channel
    alpha = img_np[:, :, 3]

    # Step 4: Create strong mask for visible pixels
    mask = (alpha > 200).astype(np.uint8)  # keep only mostly opaque pixels

    # Step 5: Apply mask and flatten onto pure white background
    rgb = img_np[:, :, :3]
    white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
    rgb = rgb * mask[:, :, None] + white_bg * (1 - mask[:, :, None])

    return Image.fromarray(rgb)

def enhance_image(pil_img):
    """Enhance sharpness"""
    return ImageEnhance.Sharpness(pil_img).enhance(2.0)

def resize_to_2x2(pil_img, dpi=300):
    """Resize to 2x2 inches at 300 dpi (600x600 px)"""
    size_px = int(2 * dpi)
    return pil_img.resize((size_px, size_px), Image.LANCZOS)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_folder():
    files = request.files.getlist("images[]")
    results = []

    for file in files:
        original_name = file.filename  # keep original filename
        try:
            # Open and ensure RGBA
            pil_img = Image.open(file.stream).convert("RGBA")
            
            # Resize very large images first (max 2000x2000)
            pil_img.thumbnail((2000, 2000))

            # Process image
            pil_img = crop_from_nose_centered(pil_img)
            pil_img = remove_bg_white(pil_img)
            pil_img = enhance_image(pil_img)
            pil_img = resize_to_2x2(pil_img)

            # Save processed image with original filename
            safe_name = original_name.replace("/", "_").replace("\\", "_")
            output_path = os.path.join(app.config["PROCESSED_FOLDER"], safe_name)
            pil_img.save(output_path)

            results.append({"original": original_name, "processed": safe_name, "success": True})
            processed_map[original_name] = safe_name

        except Exception as e:
            # Log error but continue
            results.append({
                "original": original_name,
                "processed": None,
                "success": False,
                "error": str(e)
            })

    return jsonify({"results": results})


@app.route("/processed/<filename>")
def download_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

@app.route("/download_all")
def download_all():
    zip_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed_images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for orig_name, processed_file in processed_map.items():
            file_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_file)
            if os.path.exists(file_path):
                # save in ZIP using the original filename
                zipf.write(file_path, arcname=orig_name)
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
