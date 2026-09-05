# Intialisation

## create a virtual environment and activate it

Windows\
python -m venv .venv\
.venv\Scripts\activate

## Install requirements

pip install -r requirements.txt\

# Run

## Current code functionality

# Development instructions

git --no-pager log > gitLog.txt
pip freeze > requirements.txt
## Lung CT bounding-box pipeline

See [alexNet/README.md](alexNet/README.md) for XML/DICOM matching, PNG preparation,
AlexNet-based lesion detection training, and prediction overlays.
