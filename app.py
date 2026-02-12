import os
import io
import zipfile
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageEnhance
from rembg import remove

app = Flask(__name__)

# -----------------------------
# Temporary folder for Render free plan
# -----------------------------
PROCESSED_FOLDER = "/tmp/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# -----------------------------
# Store mapping of original -> processed filename
# -----------------------------
processed_map = {}

# -----------------------------
# Image processing functions
# -----------------------------
def crop_center(pil_img):
    """Center-crop square (fallback if face/nose detection fails)"""
    w, h = pil_img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    return pil_img.crop((left, top, left + min_side, top + min_side))

def remove_bg_white(pil_img):
    """Remove background and force pure white background"""
    try:
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")
        removed_bytes = remove(img_bytes.getvalue())
        pil_img_rgba = Image.open(io.BytesIO(removed_bytes)).convert("RGBA")

        img_np = np.array(pil_img_rgba)
        alpha = img_np[:, :, 3]
        mask = (alpha > 200).astype(int)
        rgb = img_np[:, :, :3]
        white_bg = 255 * np.ones_like(rgb, dtype=np.uint8)
        rgb = rgb * mask[:, :, None] + white_bg * (1 - mask[:, :, None])
        return Image.fromarray(rgb)
    except Exception:
        return pil_img.convert("RGB")  # fallback if rembg fails

def enhance_image(pil_img):
    return ImageEnhance.Sharpness(pil_img).enhance(2.0)

def resize_to_2x2(pil_img, dpi=300):
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
        original_name = file.filename
        try:
            # Open and convert to RGBA
            pil_img = Image.open(file.stream).convert("RGBA")

            # Resize very large images to avoid memory issues
            pil_img.thumbnail((800, 800))

            # Center crop
            pil_img = crop_center(pil_img)

            # Remove background
            pil_img = remove_bg_white(pil_img)

            # Enhance and resize
            pil_img = enhance_image(pil_img)
            pil_img = resize_to_2x2(pil_img)

            # Save processed image to /tmp
            safe_name = original_name.replace("/", "_").replace("\\", "_")
            output_path = os.path.join(PROCESSED_FOLDER, safe_name)
            pil_img.save(output_path)

            # Map original -> saved file
            processed_map[original_name] = safe_name
            results.append({"original": original_name, "processed": safe_name, "success": True})

        except Exception as e:
            results.append({
                "original": original_name,
                "processed": None,
                "success": False,
                "error": str(e)
            })

    return jsonify({"results": results})

@app.route("/download_all")
def download_all():
    zip_path = os.path.join(PROCESSED_FOLDER, "processed_images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for orig_name, saved_name in processed_map.items():
            file_path = os.path.join(PROCESSED_FOLDER, saved_name)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=orig_name)
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    import numpy as np
    app.run(host="0.0.0.0", port=5000)
