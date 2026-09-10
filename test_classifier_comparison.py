import csv
import tempfile
import unittest
from pathlib import Path

from alexNet.compare_classifiers import confusion_metrics, read_rows
from alexNet.prepare_classification import slice_label


class ClassificationDataTests(unittest.TestCase):
    def test_both_heads_fit_training_only_and_evaluate_same_holdouts(self):
        import numpy as np
        from alexNet.compare_classifiers import fit_compare
        features = np.array([[-3., -2.], [-2., -1.], [2., 1.], [3., 2.],
                             [-100., -100.], [100., 100.], [-200., -200.], [200., 200.]])
        rows = [dict(patient_id=f'p{i}', image=f'{i}.png',
                     label='lesion_present' if features[i, 0] > 0 else 'lesion_absent',
                     split='train' if i < 4 else 'val' if i < 6 else 'test')
                for i in range(8)]
        models, report, predictions = fit_compare(features, rows)
        self.assertEqual(set(models), {'svm', 'random_forest'})
        scaler = models['svm'].named_steps['standardscaler']
        self.assertEqual(scaler.n_samples_seen_, 4)
        np.testing.assert_allclose(scaler.var_, features[:4].var(axis=0))
        for name in models:
            self.assertEqual(set(report['metrics'][name]), {'val', 'test'})
            self.assertEqual({p['image'] for p in predictions if p['classifier'] == name},
                             {'4.png', '5.png', '6.png', '7.png'})

    def test_confusion_metrics_counts_and_ratios(self):
        result = confusion_metrics([[8, 2], [3, 7]], ['lesion_absent', 'lesion_present'])
        self.assertEqual(result['positive_class'], 'lesion_present')
        for key, expected in dict(true_positive=7, false_positive=2,
                                  true_negative=8, false_negative=3,
                                  sensitivity=0.7, specificity=0.8, precision=7/9).items():
            self.assertAlmostEqual(result[key], expected)
        reversed_result = confusion_metrics([[7, 3], [2, 8]], ['lesion_present', 'lesion_absent'])
        self.assertEqual(result, reversed_result)

    def test_confusion_metrics_no_positive_predictions(self):
        result = confusion_metrics([[169, 0], [31, 0]], ['lesion_absent', 'lesion_present'])
        self.assertEqual((result['sensitivity'], result['specificity'], result['precision']),
                         (0.0, 1.0, 0.0))
        empty = confusion_metrics([[0, 0], [0, 0]], ['lesion_absent', 'lesion_present'])
        self.assertEqual((empty['sensitivity'], empty['specificity'], empty['precision']),
                         (0.0, 0.0, 0.0))

    def test_confusion_metrics_multiclass(self):
        result = confusion_metrics([[5, 1, 2], [3, 4, 0], [1, 2, 6]], ['a', 'b', 'c'])
        self.assertNotIn('positive_class', result)
        values = result['per_class_metrics']['b']
        self.assertEqual([values[key] for key in ('true_positive', 'false_positive',
                                                'true_negative', 'false_negative')], [4, 3, 14, 3])

    def test_labels_match_patient_and_slice(self):
        annotations = {('p1', 'uid1'): 'annotation.xml'}
        self.assertEqual(slice_label(('p1', 'uid1'), annotations), 'lesion_present')
        self.assertEqual(slice_label(('p1', 'uid2'), annotations), 'lesion_absent')
        self.assertEqual(slice_label(('p2', 'uid1'), annotations), 'lesion_absent')

    def write_csv(self, folder, patients=('p1', 'p2', 'p3'), labels=('a', 'b', 'a')):
        path = folder / 'labels.csv'
        with path.open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['image', 'patient_id', 'label', 'split'])
            for i, (patient, label, split) in enumerate(zip(patients, labels, ('train', 'train', 'test'))):
                (folder / f'{i}.png').touch()
                writer.writerow([f'{i}.png', patient, label, split])
        return path

    def test_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(self.write_csv(Path(folder)))
            self.assertEqual(rows[0]['image'], str(Path(folder).resolve() / '0.png'))

    def test_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                read_rows(self.write_csv(Path(folder), patients=('p1', 'p2', 'p1')))

    def test_unseen_class(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'absent from training'):
                read_rows(self.write_csv(Path(folder), labels=('a', 'b', 'c')))

    def test_single_class(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'at least two classes'):
                read_rows(self.write_csv(Path(folder), labels=('a', 'a', 'a')))


if __name__ == '__main__':
    unittest.main()
