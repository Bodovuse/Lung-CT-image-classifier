"""Prepare all eligible CT slices and compare AlexNet + SVM against AlexNet + Random Forest."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from alexNet.prepare_classification import prepare
from alexNet.compare_classifiers import main as compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dicoms', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--exclude-missing-patients', action='store_true')
    args = parser.parse_args()
    
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    import math
    
    if args.epochs < 1 or args.patience < 1 or not 0 < args.learning_rate < math.inf:
        parser.error('Invalid fine-tuning parameters')
    
    if args.output.exists():
        raise FileExistsError('Choose a new output directory to preserve previous results')
    
    prepare(SimpleNamespace(annotations=args.annotations, dicoms=args.dicoms,
        output=args.output / 'dataset', seed=args.seed, resume=False,
        exclude_missing_patients=args.exclude_missing_patients,
        max_patients=None, max_slices_per_patient=None))
    compare(['--labels', str(args.output / 'dataset/labels.csv'),
             '--output', str(args.output / 'comparison'), '--seed', str(args.seed),
             '--device', args.device, '--batch-size', str(args.batch_size),
             '--epochs', str(args.epochs), '--patience', str(args.patience),
             '--learning-rate', str(args.learning_rate)])


if __name__ == '__main__':
    main()
