import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from alexNet.dataset_policy import EXCLUDED_SERIES
from alexNet.prepare_classification import prepare


class PreparationTests(unittest.TestCase):
    def test_rgb_and_grayscale_labels_scan_exclusions_and_patient_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(7):
                patient = f'p{i}'
                dicoms = root / 'dicoms' / ('Lung_Dx-' + patient)
                xmls = root / 'annotations' / patient
                dicoms.mkdir(parents=True)
                xmls.mkdir(parents=True)
                for j in range(3):
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
                    ds.PatientID = 'Lung_Dx-' + patient
                    ds.SOPInstanceUID = generate_uid()
                    ds.SeriesInstanceUID = next(iter(EXCLUDED_SERIES)) if j == 2 else generate_uid()
                    ds.Modality = 'CT'
                    ds.Rows = ds.Columns = 8
                    ds.PhotometricInterpretation = 'RGB' if i % 2 else 'MONOCHROME2'
                    ds.SamplesPerPixel = 3 if i % 2 else 1
                    if i % 2:
                        ds.PlanarConfiguration = 0
                    ds.BitsAllocated = ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0
                    ds.PixelData = np.zeros((8, 8, ds.SamplesPerPixel), dtype=np.uint8).tobytes()
                    ds.save_as(dicoms / f'{j}.dcm')
                    if j != 1:
                        (xmls / (str(ds.SOPInstanceUID) + '.xml')).write_text('<annotation/>')
            prepare(SimpleNamespace(dicoms=root / 'dicoms', annotations=root / 'annotations',
                output=root / 'output', seed=42, resume=False, exclude_missing_patients=False,
                max_patients=None, max_slices_per_patient=None))
            with (root / 'output/labels.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 14)
            for patient in {r['patient_id'] for r in rows}:
                self.assertEqual(len({r['split'] for r in rows if r['patient_id'] == patient}), 1)
            for split in ('train', 'val', 'test'):
                self.assertEqual({r['label'] for r in rows if r['split'] == split},
                                 {'lesion_present', 'lesion_absent'})
            report = json.loads((root / 'output/report.json').read_text())
            self.assertEqual(report['excluded_scan_slices'], 7)
