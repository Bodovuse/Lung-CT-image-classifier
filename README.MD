# Intialisation

## create a virtual environment and activate it

Windows\
python -m venv .venv\
.venv\Scripts\activate

## Install requirements

pip install -r requirements.txt\

# Convert DICOM to PNG 
python dicomConvert.py -f "D:\Project data" "artifacts\png"

/ this is for bulk conversion for training AlexNet uses prepare_voc.py to convert that only uses slices that have matching XML annotations

# Train
python alexNet/alexNet.py train --manifest artifacts/lung_ct/manifest.json


# Predict boxes on a PNG
python alexNet/alexNet.py predict --model artifacts/lesion_detector.pt --image "path/to/scan.png"

## Current code functionality

# Development instructions

git --no-pager log > gitLog.txt
pip freeze > requirements.txt
## Lung CT bounding-box pipeline

See [alexNet/README.md](alexNet/README.md) for XML/DICOM matching, PNG preparation,
AlexNet-based lesion detection training, and prediction overlays.
