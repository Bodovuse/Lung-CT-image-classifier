# Lung CT lesion detection

`alexNet.py` now uses a PyTorch AlexNet convolutional backbone with a Faster R-CNN
head. It learns to locate annotated lesions, including multiple boxes per slice.
It replaces the earlier TensorFlow slice-classification draft. Training starts
from random weights; no pretrained cancer detector is included.

Use Python 3.12 and install `pip install -r alexNet/requirements.txt`.
The root requirements also include the detector dependencies; TensorFlow is no
longer required for this pipeline.

Prepare the annotated CT images (PowerShell, from the repository root):

```powershell
python alexNet/prepare_voc.py --annotations "C:\Users\Admin\Documents\Masters\project\Data\Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020" --dicoms "D:\Project data" --output artifacts/lung_ct
```

This reads DICOM headers to match SOPInstanceUID against the XML filename, ignores
PET images, converts only matched CT slices, and writes a manifest and a report
of unmatched annotations/read errors. Inspect the report before training. The XML
filename/path fields are placeholders in this dataset and are not used. Existing
exports can be reused with `--png-root PATH`: that folder must mirror precisely
the `--dicoms` root as produced by `dicomConvert.py -f`, with `.dcm.png` filenames.
No DICOM files or XML annotations are modified.

For a small preparation check add `--patient A0001` and choose a separate output
folder. Such a one-patient sample is for inspection, not training/validation.
Five original-size PNGs with ground-truth boxes are saved for visual inspection.
Default box conversion assumes standard one-based inclusive VOC coordinates;
use `--voc-origin 0` if your annotation source uses zero-based coordinates.
The PNGs must retain original dimensions and orientation.

```powershell
python alexNet/alexNet.py train --manifest artifacts/lung_ct/manifest.json --epochs 20 --batch-size 2 --model artifacts/lesion_detector.pt
python alexNet/alexNet.py predict --model artifacts/lesion_detector.pt --image "path/to/scan.png" --output artifacts/prediction
```

Use `--device cuda` with a compatible PyTorch CUDA installation. CPU training is
supported but slow. Predictions are written as JSON boxes in original image
coordinates and a PNG overlay. Training saves the checkpoint with the highest
validation F1 at score >= 0.5 and IoU >= 0.5 (threshold configurable). These are
box-level precision/recall/F1, not patient-level cancer accuracy or mAP. Reserve
an additional independent patient test set before reporting final performance;
do not repeatedly tune against that test set.

The preparation split is reproducible and patient-disjoint (80/20 by default).
All A/B/E/G boxes are combined into one `lesion` class. The collection contains
lung cancer cases; absent XML files are not negative cancer labels. Background
region proposals are used by the detector during training, but evaluation here
covers annotated slices only and cannot establish performance on cancer-free
patients. Scores are model outputs, not calibrated cancer probabilities.

The converter retains per-slice intensity scaling for compatibility with existing
exports, handles signed/constant arrays, and preserves spatial dimensions. These
8-bit PNGs do not retain HU calibration or 3D spacing. Use consistent conversion
for training and prediction; this is a research baseline, not a validated diagnosis
system. Lung-window preprocessing and a backbone with finer spatial features
would be separate experiments.

References: [TCIA dataset](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
and [TorchVision detection interface](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

