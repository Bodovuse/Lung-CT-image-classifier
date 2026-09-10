# CNN classification: SVM versus Random Forest

The project first fine-tunes an ImageNet-initialized AlexNet CNN on CT slice
labels, then compares **SVM and Random Forest on the same learned features**.
Both classifiers predict `lesion_present` or `lesion_absent` for a whole slice.

During fine-tuning, a temporary linear two-class neural head supplies gradients
through class-weighted cross-entropy. AdamW updates the CNN using training patients
only. Validation macro F1 selects the best checkpoint; early stopping defaults to
5 epochs without improvement, with a maximum of 20 epochs and learning rate 1e-5.
Class weights are computed from training labels only. Test images are never loaded
during fine-tuning. Labels come from XML presence, not classifier pseudo-labels.

After selection, the auxiliary head is unused and the CNN is frozen. The backbone
omits AlexNet's final max pool and uses adaptive average pooling to 6 x 6, yielding
9,216 features per slice. Both final classifiers receive identical features and
patient splits. SVM uses training-fitted standard scaling and an RBF kernel (C=1);
Random Forest uses 300 trees. Both use balanced class weights; the seed is 42.

## Setup and full comparison

Use Python 3.12 and run commands from the repository root:

```powershell
python -m pip install -r alexNet/requirements.txt
python -m alexNet.train_full_dataset --dicoms "D:/Project data/lung_pet_ct_dx" --annotations "../Data/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020" --output artifacts/full_classifier_comparison --exclude-missing-patients
```

This prepares every eligible slice, fine-tunes AlexNet, extracts shared features, fits both classifiers,
and reports validation and held-out test performance. Use a new output directory
for each run. Feature extraction downloads ImageNet weights if not already cached.
`--batch-size` controls CNN training/extraction batches; `--device cuda` enables GPU use.
Use `--epochs`, `--patience`, and `--learning-rate` to configure fine-tuning.
Partial runs retain their best CNN checkpoint and history; automatic resume is not implemented.
Full-dataset RBF SVM fitting and storing 9,216 features per slice can require
substantial memory and time; the runner does not silently sample the dataset.

## Separate preparation and comparison

```powershell
python -m alexNet.prepare_classification --dicoms "D:/Project data/lung_pet_ct_dx" --annotations "../Data/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020" --output artifacts/classification_dataset --exclude-missing-patients
python -m alexNet.compare_classifiers --labels artifacts/classification_dataset/labels.csv --output artifacts/classifier_comparison
```

Preparation applies `dataset_policy.py` patient and scan exclusions, matches
patient ID + SOPInstanceUID, and supports single-frame grayscale / unsigned 8-bit
RGB CT. Under the dataset owner's labeling rule, matching XML means lesion present;
no matching XML means lesion absent. Supply the complete annotation directory.
Grayscale uses per-slice 8-bit scaling; RGB pixels are preserved. AlexNet then uses
its ImageNet resize, center crop and normalization. Both classifiers see the same
transformed images. Patient splits are approximately 70/15/15 (220/47/47 for 314
eligible patients), with both classes required in each split.

`--max-patients` and `--max-slices-per-patient` are optional preparation flags for
pilot experiments. `--resume` reuses verified PNGs after interrupted preparation.
Prepared CSV columns are `image,patient_id,label,split`; relative image paths are
resolved against the CSV directory. Patient leakage and unseen evaluation classes
are rejected. Validation/test slices are never passed to classifier fitting or
scaler fitting. Classifier hyperparameters are fixed; CNN checkpoint selection uses validation only.

## Outputs

- `alexnet_features.pt`: selected CNN weights, auxiliary training head and selection metadata.
- `finetuning_history.json`: training loss and validation macro F1 by epoch.
- `svm.joblib` and `random_forest.joblib`: both fitted classification heads.
- `features.npz` and `rows.json`: shared feature matrix and ordered slice metadata.
- `metrics.json`: accuracy, balanced accuracy, confusion matrices, sensitivity,
  specificity, precision, and per-class F1 for validation/test.
- `predictions.csv`: per-slice predictions from both classifiers.

Export a comparison table and F1 figure with:

```powershell
python -m alexNet.export_results --metrics artifacts/classifier_comparison/metrics.json --output artifacts/classifier_report
python -m unittest discover -v
```

The current architecture is illustrated in `artifacts/model_flowchart.png`, with
editable Mermaid source alongside it. Older detector artifacts are historical,
superseded outputs and are not inputs to this classification pipeline.
