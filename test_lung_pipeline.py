import io
import unittest
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from dicomConvert import ct_to_png


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
