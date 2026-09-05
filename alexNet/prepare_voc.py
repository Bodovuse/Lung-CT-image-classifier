"""Match TCIA VOC XML to CT slices by SOPInstanceUID and prepare PNG detection data."""
import argparse
import json
import random
from pathlib import Path


def read_boxes(path, origin=1):
    from pascal_voc_tools import PascalXml
    annotation = PascalXml().load(str(path))
    width, height = annotation.size.width, annotation.size.height
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image size: {path}')
    boxes = []
    names = []
    for obj in annotation.object:
        b = obj.bndbox
        # Standard VOC: one-based inclusive -> zero-based half-open coordinates.
        box = [b.xmin - origin, b.ymin - origin, b.xmax, b.ymax]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f'Invalid box {box}: {path}')
        boxes.append(box)
        names.append(obj.name)
    if not boxes:
        raise ValueError(f'No annotated lesions: {path}; not assumed to be a negative slice')
    return width, height, boxes, names


def prepare(args):
    import pydicom
    from PIL import Image, ImageDraw
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dicomConvert import convert_file
    annotations = sorted(args.annotations.rglob('*.xml'))
    if not annotations:
        raise ValueError('No XML annotations found')
    if args.patient:
        annotations = [p for p in annotations if p.parent.name == args.patient]
    wanted = {}
    for path in annotations:
        if path.stem in wanted:
            raise ValueError(f'Duplicate annotation UID: {path.stem}')
        wanted[path.stem] = path
    if not wanted:
        raise ValueError('No annotations selected')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Output already has a manifest; choose a new output directory')
    records, matched, failures = [], set(), []
    dicom_root = args.dicoms.resolve()
    scan_root = dicom_root
    if args.patient:
        candidates = list(dicom_root.rglob('Lung_Dx-' + args.patient))
        if len(candidates) != 1:
            raise ValueError('Expected one matching patient DICOM directory')
        scan_root = candidates[0]
    for number, path in enumerate(scan_root.rglob('*.dcm'), 1):
        if number % 5000 == 0:
            print(f'Read {number} DICOM headers; matched {len(matched)} slices', flush=True)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                                  specific_tags=['SOPInstanceUID', 'Modality', 'Rows', 'Columns'])
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
            continue
        uid = str(getattr(ds, 'SOPInstanceUID', ''))
        if uid not in wanted or getattr(ds, 'Modality', '') != 'CT':
            continue
        if uid in matched:
            raise ValueError(f'Duplicate matched CT SOPInstanceUID: {uid}')
        xml = wanted[uid]
        width, height, boxes, names = read_boxes(xml, args.voc_origin)
        if (int(ds.Columns), int(ds.Rows)) != (width, height):
            raise ValueError(f'XML/DICOM size mismatch: {uid}')
        if args.png_root:
            # Exact relative path emitted by dicomConvert.py -f with this DICOM root.
            png = args.png_root.resolve() / path.relative_to(dicom_root).with_name(path.name + '.png')
            if not png.is_file():
                raise FileNotFoundError(png)
        else:
            png = (args.output / 'images' / xml.parent.name / (uid + '.png')).resolve()
            png.parent.mkdir(parents=True, exist_ok=True)
            convert_file(path, png)
        with Image.open(png) as image:
            if image.size != (width, height) or image.mode != 'L':
                raise ValueError(f'Expected original-size 8-bit grayscale PNG: {png}')
            if len(records) < 5:
                preview = image.convert('RGB')
                draw = ImageDraw.Draw(preview)
                for x1, y1, x2, y2 in boxes:
                    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline='red', width=2)
                preview.save(args.output / f'annotation_preview_{len(records)}.png')
        matched.add(uid)
        records.append(dict(image=str(png), patient_id=xml.parent.name, uid=uid,
                            boxes=boxes, source_labels=names, width=width, height=height))
    patients = sorted({r['patient_id'] for r in records})
    random.Random(args.seed).shuffle(patients)
    n_val = max(1, min(len(patients) - 1, round(len(patients) * args.val_fraction))) if len(patients) > 1 else 0
    validation = set(patients[:n_val])
    for record in records:
        record['split'] = 'val' if record['patient_id'] in validation else 'train'
    report = dict(matched=len(records), patients=len(patients),
                  unmatched_xml=[str(wanted[uid]) for uid in sorted(wanted.keys() - matched)],
                  dicom_read_failures=failures)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not records:
        raise ValueError('No matching annotated CT images; see report.json')
    manifest = dict(format_version=1, classes=['background', 'lesion'], voc_origin=args.voc_origin,
                    seed=args.seed, records=records)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} slices from {len(patients)} patients; {len(report["unmatched_xml"])} unmatched XMLs. See report.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--dicoms', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--png-root', type=Path, help='Existing converter output, mirroring --dicoms')
    parser.add_argument('--patient', help='Prepare one patient for inspection only, e.g. A0001')
    parser.add_argument('--voc-origin', type=int, choices=[0, 1], default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error('--val-fraction must be between 0 and 1')
    prepare(args)


if __name__ == '__main__':
    main()
