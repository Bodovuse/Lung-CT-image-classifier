"""Export only the SVM / Random Forest comparison to CSV and an F1 PNG."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding='utf-8-sig'))
    if set(report['metrics']) != {'svm', 'random_forest'}:
        raise ValueError('Expected results for both SVM and Random Forest')
    rows = []
    for name, splits in report['metrics'].items():
        for split, metrics in splits.items():
            row = dict(classifier=name, split=split)
            for key in ('slices', 'patients', 'accuracy', 'balanced_accuracy',
                        'sensitivity', 'specificity', 'precision', 'true_positive',
                        'false_positive', 'true_negative', 'false_negative'):
                row[key] = metrics[key]
            row['f1'] = metrics['classification_report']['lesion_present']['f1-score']
            rows.append(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    bars = ax.bar([r['classifier'] + '\n' + r['split'] for r in rows],
                  [r['f1'] for r in rows], color=['#3974ab' if r['classifier'] == 'svm' else '#288477' for r in rows])
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set(ylim=(0, 1.08), ylabel='Lesion-present F1',
           title='AlexNet features: SVM versus Random Forest')
    fig.savefig(args.output / 'f1_scores.png', dpi=180)
    plt.close(fig)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
