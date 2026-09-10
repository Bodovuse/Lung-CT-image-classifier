"""Review the matching audit's completed caches without interrupting its scan.

Run again after investigate_matching.py completes for a full inventory. Missing
UIDs are only definitive within completed patient caches; cross-patient searches
remain provisional until all patients have been scanned.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from alexNet.dataset_policy import EXCLUDED_PATIENTS
from alexNet.investigate_matching import DICOM, OUT, ROOT, write


def main():
    prior = json.loads((ROOT / 'artifacts/full_detector_70pct_20260906/preparation_report.json').read_text())
    wanted = {Path(p).stem for p in prior['unmatched_xml']}
    folders = {p.name for p in DICOM.glob('Lung_Dx-*') if p.is_dir()}
    caches = sorted(p for p in (OUT / 'headers').glob('*.json') if p.stem in folders)
    scanned = {p.stem for p in caches}
    direct, meta, references = (defaultdict(list) for _ in range(3))
    series_bytes, series_images, modalities = Counter(), Counter(), Counter()
    read_errors, identity_errors, duplicates = [], [], []
    identity_pairs = Counter()
    seen = set()
    stale_caches = []
    for cache in caches:
        data = json.loads(cache.read_text())
        # A nested patient directory may have been moved while the audit ran.
        if any(r['patient_id'].removeprefix('Lung_Dx-') != r['folder_patient']
               and not Path(r['path']).exists() for r in data['records']):
            stale_caches.append(str(cache))
            scanned.remove(cache.stem)
            continue
        read_errors.extend(data['errors'])
        for r in data['records']:
            series_bytes[r['series']] += r['bytes']
            series_images[r['series']] += 1
            modalities[r['modality']] += 1
            if r['patient_id'].removeprefix('Lung_Dx-') != r['folder_patient']:
                identity_errors.append(r['path'])
                identity_pairs[(r['folder_patient'], r['patient_id'])] += 1
            if r['uid'] and r['uid'] in seen:
                duplicates.append(r['path'])
            seen.add(r['uid'])
            if r['uid'] in wanted:
                direct[r['uid']].append(r)
            if r['file_meta_uid'] in wanted:
                meta[r['file_meta_uid']].append(r['path'])
            for uid in r['references']:
                if uid in wanted:
                    references[uid].append(r['path'])
    inventory = defaultdict(Counter)
    series_issues = []
    with Path('D:/Project data/metadata/metadata.csv').open(newline='', encoding='utf-8-sig') as source:
        for r in csv.DictReader(source):
            patient, series = r['PatientID'], r['SeriesInstanceUID']
            state = ('bytes_equal' if series_images[series] and series_bytes[series] == int(r['FileSize']) else
                     'bytes_differ' if series_images[series] else
                     'patient_folder_absent' if patient not in folders else
                     'scan_pending' if patient not in scanned else
                     'series_absent')
            inventory[patient.removeprefix('Lung_Dx-')][state] += 1
            if state != 'bytes_equal':
                series_issues.append(dict(patient=patient, series=series, state=state,
                    expected_bytes=int(r['FileSize']), local_bytes=series_bytes[series],
                    download_status=r['completion_status']))
    results = []
    for name in prior['unmatched_xml']:
        path = Path(name)
        patient, uid = path.parent.name, path.stem
        matches = direct[uid]
        same = [r for r in matches if r['folder_patient'] == patient]
        state = ('matched_same_patient' if same else 'matched_different_patient' if matches else
                 'patient_folder_absent' if 'Lung_Dx-' + patient not in folders else
                 'scan_pending' if 'Lung_Dx-' + patient not in scanned else 'uid_absent_in_patient')
        result = dict(xml=name, patient=patient, uid=uid, state=state,
            excluded_patient=patient in EXCLUDED_PATIENTS, matches=matches,
            file_meta_matches=meta[uid], referenced_by=references[uid])
        for match in matches:
            actual_patient = match['patient_id'].removeprefix('Lung_Dx-')
            counterpart = path.parent.parent / actual_patient / path.name
            match['actual_patient_excluded'] = actual_patient in EXCLUDED_PATIENTS
            match['xml_under_dicom_patient'] = str(counterpart) if counterpart.exists() else None
            match['xml_bytes_identical'] = counterpart.read_bytes() == path.read_bytes() if counterpart.exists() else None
            if counterpart.exists():
                try:
                    def boxes(p):
                        return [[o.findtext('bndbox/' + f) for f in ('xmin', 'ymin', 'xmax', 'ymax')]
                                for o in ET.parse(p).getroot().findall('object')]
                    match['xml_box_coordinates_identical'] = boxes(path) == boxes(counterpart)
                except ET.ParseError:
                    match['xml_box_coordinates_identical'] = None
        try:
            tree = ET.parse(path).getroot()
            result['image_fields'] = {f: tree.findtext(f, '') for f in ('folder', 'filename', 'path')}
        except ET.ParseError as exc:
            result['parse_error'] = str(exc)
            result['first_bytes_hex'] = path.read_bytes()[:32].hex()
        results.append(result)
    patients = []
    for patient in sorted({r['patient'] for r in results}):
        group = [r for r in results if r['patient'] == patient]
        patients.append(dict(patient=patient, prior_unmatched=len(group),
            states=dict(Counter(r['state'] for r in group)), inventory=dict(inventory[patient])))
    summary = dict(scanned_patients=len(scanned), current_patient_folders=len(folders),
        stale_caches_skipped=stale_caches,
        eligible_patient_folders=len(folders - {'Lung_Dx-' + p for p in EXCLUDED_PATIENTS}),
        scan_complete=scanned == folders, indexed_images=sum(modalities.values()),
        modalities=dict(modalities), read_errors=len(read_errors),
        patient_identity_errors=identity_errors, duplicate_uid_paths=duplicates,
        patient_identity_conflicts=[dict(folder=a, header_patient=b, images=n)
                                    for (a, b), n in identity_pairs.items()],
        prior_unmatched=len(results), states=dict(Counter(r['state'] for r in results)),
        file_meta_matches=sum(bool(r['file_meta_matches']) for r in results),
        referenced_matches=sum(bool(r['referenced_by']) for r in results),
        parse_errors=[r for r in results if 'parse_error' in r],
        inventory=dict(sum(inventory.values(), Counter())), patients=patients,
        size_difference_bytes=dict(Counter(r['local_bytes'] - r['expected_bytes']
            for r in series_issues if r['state'] == 'bytes_differ')),
        limitations=['Header caches are snapshots and are not automatically invalidated after source changes.',
                      'Equal byte totals do not prove file integrity or completeness of the source inventory.',
                      'Cross-patient absence is provisional until scan_complete is true.'])
    write(OUT / 'followup_summary.json', summary)
    write(OUT / 'followup_unmatched.json', results)
    write(OUT / 'followup_series_issues.json', series_issues)
    print(json.dumps({k: v for k, v in summary.items() if k not in
        ('patients', 'parse_errors', 'patient_identity_errors', 'duplicate_uid_paths')}, indent=2))


if __name__ == '__main__':
    main()
