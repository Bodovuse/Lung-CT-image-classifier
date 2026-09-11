"""Prepare slice labels using this dataset's XML-present / XML-absent rule."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from alexNet.dataset_policy import EXCLUDED_PATIENTS, eligible_patient_folders
from alexNet.dataset_policy import EXCLUDED_SERIES, EXCLUDED_ANNOTATION_UIDS, SCAN_POLICY_DIGEST


def slice_label(uid, annotations):
    return 'lesion_present' if uid in annotations else 'lesion_absent'


def prepare(args):
    import pydicom
    from dicomConvert import convert_file, supported_pixel_format
    annotations = {}

    for xml in args.annotations.rglob('*.xml'):
        if xml.stem in EXCLUDED_ANNOTATION_UIDS:
            continue

        key = (xml.parent.name, xml.stem)

        if key in annotations:
            raise ValueError(f'Duplicate XML UID: {xml.stem}')
        
        annotations[key] = xml

    if not annotations:
        raise ValueError('No XML annotations found')
    
    if args.output.exists() and not args.resume:
        raise FileExistsError('Choose a new output directory')
    
    # Restrict to the dataset patient folders; unrelated files are not negatives.
    folders = sorted(args.dicoms.glob('Lung_Dx-*'))
    if not folders:
        folders = sorted(args.dicoms.rglob('Lung_Dx-*'))

    folders = eligible_patient_folders([p for p in folders if p.is_dir()])
    annotations = {key: xml for key, xml in annotations.items() if key[0] not in EXCLUDED_PATIENTS}
    patients = [p.name.removeprefix('Lung_Dx-') for p in folders]

    if len(set(patients)) != len(patients) or len(patients) < 3:
        raise ValueError('Expected at least three unique Lung_Dx patient directories')
    
    missing_patients = {patient for patient, uid in annotations} - set(patients)
    
    if missing_patients and not args.exclude_missing_patients:
        raise ValueError('Annotation patients missing DICOM folders: ' + ', '.join(sorted(missing_patients)))
    
    excluded_annotations = sum(patient in missing_patients for patient, uid in annotations)
    annotations = {key: xml for key, xml in annotations.items() if key[0] not in missing_patients}
    
    if missing_patients:
        print('Excluded unavailable patients: ' + ', '.join(sorted(missing_patients)), flush=True)
    
    if args.max_patients:
        folders = sorted(random.Random(args.seed).sample(folders, min(args.max_patients, len(folders))))
        patients = [p.name.removeprefix('Lung_Dx-') for p in folders]
        annotations = {key: xml for key, xml in annotations.items() if key[0] in patients}
   
    shuffled = sorted(patients)
    random.Random(args.seed).shuffle(shuffled)
    count = max(1, round(len(shuffled) * 0.15))
    test, val = set(shuffled[:count]), set(shuffled[count:2 * count])
    rows, seen, unsupported, unsupported_keys = [], set(), [], set()
    excluded_scan_keys = set()
    
    for number, (folder, patient) in enumerate(zip(folders, patients), 1):
        for path in sorted(folder.rglob('*.dcm')):
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                                specific_tags=['SOPInstanceUID', 'Modality', 'Rows', 'Columns',
                                               'PhotometricInterpretation', 'NumberOfFrames',
                                               'SamplesPerPixel', 'BitsAllocated', 'BitsStored',
                                               'PixelRepresentation', 'PatientID', 'SeriesInstanceUID'])
            
            if getattr(ds, 'Modality', '') != 'CT':
                continue
            
            if str(getattr(ds, 'PatientID', '')).removeprefix('Lung_Dx-') != patient:
                raise ValueError(f'DICOM patient/folder mismatch: {path}')
            
            if str(getattr(ds, 'SeriesInstanceUID', '')) in EXCLUDED_SERIES:
                excluded_scan_keys.add((patient, str(getattr(ds, 'SOPInstanceUID', ''))))
                continue
            
            if not supported_pixel_format(ds):
                unsupported.append(str(path))
                unsupported_keys.add((patient, str(getattr(ds, 'SOPInstanceUID', ''))))
                continue
            
            uid = str(getattr(ds, 'SOPInstanceUID', ''))
            key = (patient, uid)
            
            if not uid or key in seen:
                raise ValueError(f'Missing or duplicate CT UID: {path}')
            
            seen.add(key)
            rows.append(dict(source=str(path.resolve()), patient_id=patient, uid=uid,
                             label=slice_label(key, annotations),
                             split='test' if patient in test else 'val' if patient in val else 'train'))
        
        print(f'Indexed {number}/{len(folders)} patients; {len(rows)} CT slices', flush=True)
    
    missing = set(annotations) - seen - unsupported_keys - excluded_scan_keys
    
    if missing:
        raise ValueError(f'{len(missing)} XML annotations have no matching CT; check dataset roots')
    
    indexed_slices = len(rows)
    empty_patients = set(patients) - {r['patient_id'] for r in rows}
    
    if empty_patients:
        raise ValueError('Patients have no usable CT slices: ' + ', '.join(sorted(empty_patients)))
    if args.max_slices_per_patient:
        rng = random.Random(args.seed)
        selected = []
        for patient in patients:
            group = [r for r in rows if r['patient_id'] == patient]
            selected.extend(rng.sample(group, min(args.max_slices_per_patient, len(group))))
        rows = selected
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {'lesion_present', 'lesion_absent'}:
            raise ValueError(f'{split} must contain both classes')
    
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / 'selected_sources.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    
    for i, row in enumerate(rows, 1):
        png = args.output / 'images' / row['patient_id'] / (row['uid'] + '.png')
        png.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and png.exists():
            from PIL import Image
            with Image.open(png) as image:
                if image.format != 'PNG' or image.mode not in ('L', 'RGB'):
                    raise ValueError(f'Invalid existing PNG: {png}')
                image.verify()
        else:
            convert_file(row['source'], png)
        row['image'] = png.relative_to(args.output).as_posix()
        if i % 1000 == 0:
            print(f'Converted {i}/{len(rows)} CT slices', flush=True)

    with (args.output / 'labels.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['image', 'patient_id', 'label', 'split'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    
    report = dict(seed=args.seed, label_rule='User-confirmed: matching XML = lesion_present; no XML = lesion_absent',
                  scan_policy_digest=SCAN_POLICY_DIGEST, excluded_scan_slices=len(excluded_scan_keys),
                  excluded_patients=sorted(missing_patients | EXCLUDED_PATIENTS), excluded_annotations=excluded_annotations,
                  pilot=bool(args.max_patients or args.max_slices_per_patient),
                  max_patients=args.max_patients, max_slices_per_patient=args.max_slices_per_patient,
                  indexed_slices=indexed_slices,
                  unsupported_images=unsupported,
                  unsupported_annotations=len(set(annotations) & unsupported_keys),
                  annotation_root=str(args.annotations.resolve()), dicom_root=str(args.dicoms.resolve()),
                  counts=dict(Counter(r['split'] + '/' + r['label'] for r in rows)),
                  patients={s: len({r['patient_id'] for r in rows if r['split'] == s})
                            for s in ('train', 'val', 'test')})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    
    print(json.dumps({k: v for k, v in report.items() if k != 'unsupported_images'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--dicoms', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Reuse verified PNGs from an interrupted preparation')
    parser.add_argument('--exclude-missing-patients', action='store_true',
                        help='Exclude annotation patients whose DICOM folders are unavailable')
    parser.add_argument('--max-patients', type=int, help='Seeded patient subset for a pilot run')
    parser.add_argument('--max-slices-per-patient', type=int, help='Uniform slice sample per patient for a pilot run')
    args = parser.parse_args()
    
    if args.max_patients is not None and args.max_patients < 3:
        parser.error('--max-patients must be at least 3')
    if args.max_slices_per_patient is not None and args.max_slices_per_patient < 1:
        parser.error('--max-slices-per-patient must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
