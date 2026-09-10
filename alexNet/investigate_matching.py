"""Read-only image/annotation matching audit; caches DICOM headers for review."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import pydicom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/image_annotation_matching'
DICOM = Path('D:/Project data/lung_pet_ct_dx')
XML = ROOT.parent / 'Data/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def scan(folder):
    cache = OUT / 'headers' / (folder.name + '.json')
    if cache.exists():
        return json.loads(cache.read_text())
    records, errors = [], []
    for path in folder.rglob('*'):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=[
                'SOPInstanceUID', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
                'Modality', 'Rows', 'Columns', 'PhotometricInterpretation', 'NumberOfFrames',
                'SourceImageSequence', 'ReferencedImageSequence'])
            references = []
            for sequence in ('SourceImageSequence', 'ReferencedImageSequence'):
                for item in getattr(ds, sequence, []):
                    references.append(str(getattr(item, 'ReferencedSOPInstanceUID', '')))
            records.append(dict(path=str(path), folder_patient=folder.name.removeprefix('Lung_Dx-'),
                patient_id=str(getattr(ds, 'PatientID', '')), uid=str(getattr(ds, 'SOPInstanceUID', '')),
                file_meta_uid=str(getattr(ds.file_meta, 'MediaStorageSOPInstanceUID', '')),
                series=str(getattr(ds, 'SeriesInstanceUID', '')), study=str(getattr(ds, 'StudyInstanceUID', '')),
                modality=str(getattr(ds, 'Modality', '')), width=int(getattr(ds, 'Columns', 0)),
                height=int(getattr(ds, 'Rows', 0)), photo=str(getattr(ds, 'PhotometricInterpretation', '')),
                frames=int(getattr(ds, 'NumberOfFrames', 1)), references=references, bytes=path.stat().st_size))
        except Exception as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    result = dict(records=records, errors=errors)
    write(cache, result)
    return result


def main():
    (OUT / 'headers').mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in DICOM.glob('Lung_Dx-*') if p.is_dir())
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(scan, folder) for folder in folders]
        for completed, future in enumerate(as_completed(futures), 1):
            data = future.result()
            records.extend(data['records'])
            errors.extend(data['errors'])
            status = dict(state='scanning', completed_patients=completed, total_patients=len(folders),
                          images=len(records), errors=len(errors), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
            write(OUT / 'status.json', status)
            if completed % 10 == 0:
                print(json.dumps(status), flush=True)
    by_uid, by_meta, by_name, by_reference, by_series = (defaultdict(list) for _ in range(5))
    for record in records:
        by_uid[record['uid']].append(record)
        by_meta[record['file_meta_uid']].append(record)
        by_name[Path(record['path']).name].append(record)
        by_series[record['series']].append(record)
        for uid in record['references']:
            by_reference[uid].append(record)
    results = []
    for path in sorted(XML.rglob('*.xml')):
        patient, uid = path.parent.name, path.stem
        matches = by_uid.get(uid, [])
        same = [r for r in matches if r['folder_patient'] == patient]
        usable = [r for r in same if r['modality'] == 'CT' and r['photo'] in ('MONOCHROME1', 'MONOCHROME2') and r['frames'] == 1]
        category = ('matched_grayscale_ct' if usable else 'matched_other_ct' if any(r['modality'] == 'CT' for r in same)
                    else 'matched_other_modality' if same else 'matched_different_patient' if matches
                    else 'patient_folder_absent' if patient not in {p.name.removeprefix('Lung_Dx-') for p in folders}
                    else 'uid_not_found')
        result = dict(xml=str(path), patient=patient, uid=uid, category=category,
                      matches=[dict(path=r['path'], patient=r['folder_patient'], modality=r['modality'],
                                    photo=r['photo'], width=r['width'], height=r['height']) for r in matches])
        if not same:
            result['file_meta_matches'] = [r['path'] for r in by_meta.get(uid, [])]
            result['referenced_by_images'] = [r['path'] for r in by_reference.get(uid, [])]
            try:
                tree = ET.parse(path).getroot()
                fields = {field: tree.findtext(field, '') for field in ('folder', 'filename', 'path')}
                result['xml_image_fields'] = fields
                alternatives = set()
                for field in ('filename', 'path'):
                    filename = fields[field].replace('\\', '/').split('/')[-1]
                    for r in by_name.get(filename, []) + by_uid.get(Path(filename).stem, []):
                        alternatives.add(r['path'])
                result['alternate_filename_candidates'] = sorted(alternatives)
            except Exception as exc:
                result['parse_error'] = str(exc)
                result['first_bytes_hex'] = path.read_bytes()[:32].hex()
        results.append(result)
    metadata = []
    with Path('D:/Project data/metadata/metadata.csv').open(newline='', encoding='utf-8-sig') as source:
        for row in csv.DictReader(source):
            local = by_series.get(row['SeriesInstanceUID'], [])
            expected_bytes = int(row['FileSize'])
            actual_bytes = sum(r['bytes'] for r in local)
            metadata.append(dict(patient=row['PatientID'], series=row['SeriesInstanceUID'],
                expected_bytes=expected_bytes, local_bytes=actual_bytes, images=len(local),
                bytes_equal=expected_bytes == actual_bytes, download_status=row['completion_status'],
                expected_path=row['S5cmdManifestPath']))
    old = json.loads((ROOT / 'artifacts/full_detector_70pct_20260906/preparation_report.json').read_text())
    old_missing = set(old['unmatched_xml'])
    previous = [r for r in results if r['xml'] in old_missing]
    summary = dict(total_patients=len(folders), indexed_images=len(records), read_errors=len(errors),
        modalities=dict(Counter(r['modality'] for r in records)), xml_categories=dict(Counter(r['category'] for r in results)),
        prior_791_categories=dict(Counter(r['category'] for r in previous)),
        prior_791_parse_errors=sum('parse_error' in r for r in previous),
        prior_791_file_meta_matches=sum(bool(r.get('file_meta_matches')) for r in previous),
        prior_791_referenced_matches=sum(bool(r.get('referenced_by_images')) for r in previous),
        prior_791_alternate_filename_candidates=sum(bool(r.get('alternate_filename_candidates')) for r in previous),
        metadata_series=len(metadata), missing_series=sum(not r['images'] for r in metadata),
        size_mismatched_present_series=sum(bool(r['images']) and not r['bytes_equal'] for r in metadata),
        exact_size_series=sum(r['bytes_equal'] for r in metadata),
        missing_series_by_patient=dict(Counter(r['patient'] for r in metadata if not r['images'])),
        different_patient_xml=[r for r in results if r['category'] == 'matched_different_patient'],
        file_meta_uid_disagreements=sum(r['uid'] != r['file_meta_uid'] for r in records))
    write(OUT / 'xml_matches.json', results)
    write(OUT / 'series_download_audit.json', metadata)
    write(OUT / 'read_errors.json', errors)
    write(OUT / 'summary.json', summary)
    write(OUT / 'status.json', dict(state='complete', images=len(records), patients=len(folders)))
    print(json.dumps({k: v for k, v in summary.items() if k != 'different_patient_xml'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
