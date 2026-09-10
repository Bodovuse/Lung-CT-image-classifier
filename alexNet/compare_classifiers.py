"""Fine-tune AlexNet on CT labels, then compare SVM and Random Forest on shared features."""
import argparse
import csv
import json
from pathlib import Path

from alexNet.alexNet import build_backbone, load_image


def read_rows(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if not {'image', 'patient_id', 'label', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires image,patient_id,label,split columns')
        rows = list(reader)
    patients, images = {}, set()
    for row in rows:
        for key in ('image', 'patient_id', 'label', 'split'):
            row[key] = (row[key] or '').strip()
            if not row[key]:
                raise ValueError(f'Empty {key}')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        patient = row['patient_id']
        if patient in patients and patients[patient] != row['split']:
            raise ValueError(f'Patient leakage: {patient}')
        patients[patient] = row['split']
        image = (path.parent / row['image']).resolve()
        if not image.is_file() or image in images:
            raise ValueError(f'Missing or duplicate image: {image}')
        images.add(image)
        row['image'] = str(image)
    classes = {r['label'] for r in rows if r['split'] == 'train'}
    if len(classes) < 2:
        raise ValueError('Training requires at least two classes')
    if not any(r['split'] in ('val', 'test') for r in rows):
        raise ValueError('Provide a patient-disjoint val or test split')
    if {r['label'] for r in rows} - classes:
        raise ValueError('Evaluation contains classes absent from training')
    return rows


def extract_features(rows, device, batch_size, backbone=None):
    import numpy as np
    import torch
    from torchvision.models import AlexNet_Weights
    weights = AlexNet_Weights.IMAGENET1K_V1
    if backbone is None:
        backbone = build_backbone(weights)
    backbone = backbone.to(device).eval()
    backbone.requires_grad_(False)
    transform = weights.transforms()
    features = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = torch.stack([transform(load_image(r['image']))
                                 for r in rows[start:start + batch_size]]).to(device)
            output = torch.nn.functional.adaptive_avg_pool2d(backbone(batch), (6, 6))
            features.append(output.flatten(1).cpu().numpy())
            print(f'Extracted {min(start + batch_size, len(rows))}/{len(rows)} slices', flush=True)
    return np.concatenate(features)


def confusion_metrics(matrix, classes):
    """One-versus-rest counts; rows are actual, columns are predicted.

    Undefined ratios are reported as 0.0, matching classification_report.
    """
    total = sum(sum(row) for row in matrix)
    per_class = {}
    for index, label in enumerate(classes):
        tp = int(matrix[index][index])
        fn = int(sum(matrix[index]) - tp)
        fp = int(sum(row[index] for row in matrix) - tp)
        tn = int(total - tp - fn - fp)
        per_class[label] = dict(
            true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
            sensitivity=tp / (tp + fn) if tp + fn else 0.0,
            specificity=tn / (tn + fp) if tn + fp else 0.0,
            precision=tp / (tp + fp) if tp + fp else 0.0)
    result = dict(per_class_metrics=per_class, zero_division=0.0)
    if 'lesion_present' in per_class:
        result.update(positive_class='lesion_present', **per_class['lesion_present'])
    return result


def fit_compare(features, rows, seed=42):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    labels = np.array([r['label'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    train = splits == 'train'
    classes = sorted(set(labels[train]))
    models = {
        'svm': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', class_weight='balanced')),
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                               random_state=seed, n_jobs=-1),
    }
    metrics, predictions = {}, []
    for name, model in models.items():
        print(f'Fitting {name} on {int(train.sum())} training slices', flush=True)
        model.fit(features[train], labels[train])
        metrics[name] = {}
        for split in ('val', 'test'):
            mask = splits == split
            if not mask.any():
                continue
            predicted = model.predict(features[mask])
            print(f'{name} {split}: accuracy={accuracy_score(labels[mask], predicted):.2%}', flush=True)
            metrics[name][split] = dict(
                accuracy=float(accuracy_score(labels[mask], predicted)),
                balanced_accuracy=float(balanced_accuracy_score(labels[mask], predicted)),
                confusion_matrix=confusion_matrix(labels[mask], predicted, labels=classes).tolist(),
                classification_report=classification_report(labels[mask], predicted, labels=classes,
                                                             output_dict=True, zero_division=0),
                slices=int(mask.sum()),
                patients=len({rows[i]['patient_id'] for i in np.flatnonzero(mask)}))
            metrics[name][split].update(confusion_metrics(
                metrics[name][split]['confusion_matrix'], classes))
            for i, prediction in zip(np.flatnonzero(mask), predicted):
                predictions.append(dict(classifier=name, **rows[i], prediction=str(prediction)))
    return models, dict(classes=classes, metrics=metrics), predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True, help='CSV: image,patient_id,label,split')
    parser.add_argument('--output', type=Path, default=Path('artifacts/classifier_comparison'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    import math
    if args.epochs < 1 or args.patience < 1 or not 0 < args.learning_rate < math.inf:
        parser.error('Invalid fine-tuning parameters')
    rows = read_rows(args.labels)
    from alexNet.finetune import fine_tune, validate_splits
    validate_splits(rows)
    if args.output.exists():
        raise FileExistsError('Choose a new output directory to preserve previous results')
    import joblib
    # Fail before CNN extraction if the classifier's native dependencies cannot load.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    backbone, training = fine_tune(rows, args.output, args.device, args.batch_size,
                                   args.epochs, args.patience, args.learning_rate, args.seed)
    features = extract_features(rows, args.device, args.batch_size, backbone)
    np.savez_compressed(args.output / 'features.npz', features=features)
    models, report, predictions = fit_compare(features, rows, args.seed)
    report.update(feature_extractor='CT-fine-tuned AlexNet convolutions, frozen after validation selection, 6x6 average pooling',
                  fine_tuning=training, feature_checkpoint='alexnet_features.pt',
                  feature_dimensions=int(features.shape[1]), seed=args.seed,
                  evaluation_unit='CT slice', preprocessing='AlexNet_Weights.IMAGENET1K_V1.transforms()')
    for name, model in models.items():
        joblib.dump(model, args.output / f'{name}.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'predictions.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for name, splits in report['metrics'].items():
        for split, metrics in splits.items():
            print(f'{name} {split}: accuracy={metrics["accuracy"]:.2%}, '
                  f'balanced accuracy={metrics["balanced_accuracy"]:.2%}')
            for label, values in metrics['per_class_metrics'].items():
                print(f'  {label}: TP={values["true_positive"]}, FP={values["false_positive"]}, '
                      f'TN={values["true_negative"]}, FN={values["false_negative"]}, '
                      f'sensitivity={values["sensitivity"]:.2%}, '
                      f'specificity={values["specificity"]:.2%}, precision={values["precision"]:.2%}')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
