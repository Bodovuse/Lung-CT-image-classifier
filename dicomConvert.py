"""Convert single-frame grayscale or 8-bit RGB DICOMs to original-size PNGs."""
import argparse
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image


def supported_pixel_format(ds):
    if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
        return False
    photo = getattr(ds, 'PhotometricInterpretation', '')
    if photo in ('MONOCHROME1', 'MONOCHROME2'):
        return int(getattr(ds, 'SamplesPerPixel', 1)) == 1
    return (photo == 'RGB' and int(getattr(ds, 'SamplesPerPixel', 0)) == 3
            and int(getattr(ds, 'BitsAllocated', 0)) == 8
            and int(getattr(ds, 'BitsStored', 0)) == 8
            and int(getattr(ds, 'PixelRepresentation', -1)) == 0)


def ct_to_png(ct_file, png_file):
    ds = pydicom.dcmread(ct_file)
    if not supported_pixel_format(ds):
        raise ValueError('Expected a single-frame grayscale or unsigned 8-bit RGB DICOM')
    pixels = ds.pixel_array
    if ds.PhotometricInterpretation == 'RGB':
        if pixels.shape != (int(ds.Rows), int(ds.Columns), 3) or pixels.dtype != np.uint8:
            raise ValueError('Expected RGB pixels with shape (rows, columns, 3) and dtype uint8')
        Image.fromarray(pixels).save(png_file, format='PNG')
        return
    if pixels.ndim != 2:
        raise ValueError('Expected a single-frame grayscale pixel array')
    pixels = pixels.astype(np.float32)
    # Retain legacy per-slice scaling for nonnegative pixels; handle signed/constant data.
    low, high = min(0.0, float(pixels.min())), float(pixels.max())
    scaled = np.zeros_like(pixels, dtype=np.uint8) if high <= low else ((pixels - low) / (high - low) * 255).clip(0, 255).astype(np.uint8)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        scaled = 255 - scaled
    Image.fromarray(scaled).save(png_file, format='PNG')


def convert_file(ct_file_path, png_file_path):
    # Exclusive creation preserves existing exports; remove partial output on failure.
    with open(ct_file_path, 'rb') as source:
        with open(png_file_path, 'xb') as target:
            try:
                ct_to_png(source, target)
            except Exception:
                target.close()
                Path(png_file_path).unlink()
                raise


def convert_folder(ct_folder, png_folder):
    source, destination = Path(ct_folder), Path(png_folder)
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.rglob('*.dcm'):
        output = destination / path.relative_to(source).with_name(path.name + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        convert_file(path, output)
        print(f'SUCCESS {path} --> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', action='store_true')
    parser.add_argument('dicom_path')
    parser.add_argument('png_path')
    args = parser.parse_args()
    (convert_folder if args.f else convert_file)(args.dicom_path, args.png_path)
