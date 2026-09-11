"""Patient exclusions."""
import hashlib
import json
from pathlib import Path

_SCAN_POLICY = Path(__file__).with_name('scan_exclusions.json').read_bytes()
SCAN_POLICY_DIGEST = hashlib.sha256(_SCAN_POLICY).hexdigest()
_SCAN_RULES = json.loads(_SCAN_POLICY.decode('utf-8-sig'))
EXCLUDED_SERIES = frozenset(r['series_uid'] for r in _SCAN_RULES['excluded_series'])
EXCLUDED_ANNOTATION_UIDS = frozenset(_SCAN_RULES['conflicting_annotation_uids'])

EXCLUDED_PATIENTS = frozenset({'A0058', 'A0251'} | set(_SCAN_RULES.get('excluded_patients', [])))


def eligible_patient_folders(folders):
    return [folder for folder in folders
            if folder.name.removeprefix('Lung_Dx-') not in EXCLUDED_PATIENTS]
