"""Run with python -m unittest test_lung_pipeline -v."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from dicomConvert import ct_to_png
from alexNet.prepare_voc import read_boxes
from alexNet.alexNet import read_manifest


class DataTests(unittest.TestCase):
    def test_signed_and_constant_non_square_conversion(self):
        for pixels in (np.array([[-100, 0, 100], [50, -50, 25]], dtype=np.int16),
                       np.zeros((2, 3), dtype=np.int16)):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
            ds.Rows, ds.Columns = pixels.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.PixelData = pixels.tobytes()
            source, output = io.BytesIO(), io.BytesIO()
            ds.save_as(source)
            source.seek(0)
            ct_to_png(source, output)
            output.seek(0)
            image = Image.open(output)
            self.assertEqual(image.size, (3, 2))
            self.assertEqual(image.mode, 'L')
            self.assertEqual(np.asarray(image).min(), 0)
            self.assertEqual(np.asarray(image).max(), 255 if pixels.max() else 0)

    def test_voc_coordinates_and_multiple_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.xml'
            obj = '<object><name>A</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object>'
            path.write_text('<annotation><size><width>10</width><height>20</height></size>' + obj * 2 + '</annotation>')
            width, height, boxes, names = read_boxes(path)
            self.assertEqual((width, height), (10, 20))
            self.assertEqual(boxes, [[0, 1, 10, 20]] * 2)
            self.assertEqual(names, ['A', 'A'])
            path.write_text(path.read_text().replace('<xmax>10', '<xmax>11'))
            with self.assertRaises(ValueError):
                read_boxes(path)

    def test_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for index, split in enumerate(('train', 'val')):
                image = folder / f'{index}.png'
                Image.new('L', (10, 10)).save(image)
                rows.append(dict(image=str(image), patient_id='A0001', split=split,
                                 width=10, height=10, boxes=[[0, 0, 5, 5]]))
            path = folder / 'manifest.json'
            path.write_text(json.dumps(dict(format_version=1, classes=['background', 'lesion'], records=rows)))
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                read_manifest(path)


if __name__ == '__main__':
    unittest.main()
