import io

import pytest
from PIL import Image

from app.services import images


def test_accepts_jpeg(jpeg_bytes):
    ext, w, h = images.validate_upload(jpeg_bytes)
    assert (ext, w, h) == ("jpg", 320, 240)


def test_rejects_non_image():
    with pytest.raises(images.ImageError):
        images.validate_upload(b"MZ\x90\x00 definitely not an image")


def test_rejects_empty():
    with pytest.raises(images.ImageError):
        images.validate_upload(b"")


def test_rejects_disallowed_format():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="BMP")
    with pytest.raises(images.ImageError, match="JPG, PNG or WEBP"):
        images.validate_upload(buf.getvalue())


def test_rejects_tiny_image():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="PNG")
    with pytest.raises(images.ImageError, match="too small"):
        images.validate_upload(buf.getvalue())


def test_exif_orientation_and_resize():
    img = Image.new("RGB", (400, 200), (10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    full, work = images.load_for_processing(buf.getvalue(), max_side=100)
    assert full.size == (200, 400)
    assert max(work.size) <= 100 and work.size[0] % 2 == 0


def test_png_alpha_flattened():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    full, _ = images.load_for_processing(buf.getvalue(), 64)
    assert full.mode == "RGB" and full.getpixel((5, 5)) == (255, 255, 255)
