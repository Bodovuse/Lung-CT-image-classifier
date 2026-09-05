"""Lesion detection on CT PNGs using an AlexNet backbone and Faster R-CNN."""
import argparse
import json
import math
from pathlib import Path


def build_model():
    from torchvision.models import alexnet
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    # Omit the final pool to retain stride-16 spatial features. No weight download.
    backbone = alexnet(weights=None).features[:-1]
    backbone.out_channels = 256
    return FasterRCNN(backbone, num_classes=2, min_size=512, max_size=512,
                      rpn_anchor_generator=AnchorGenerator(((16, 32, 64, 128, 256),), ((0.5, 1.0, 2.0),)),
                      box_roi_pool=MultiScaleRoIAlign(['0'], output_size=7, sampling_ratio=2),
                      image_mean=[0.5] * 3, image_std=[0.5] * 3)


def load_image(path):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'L':
            raise ValueError(f'Expected an 8-bit grayscale PNG: {path}')
        return to_tensor(image.convert('RGB'))


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format_version') != 1 or data.get('classes') != ['background', 'lesion']:
        raise ValueError('Unsupported detection manifest')
    patients, seen = {}, set()
    for row in data['records']:
        patient, split = row['patient_id'], row['split']
        if split not in ('train', 'val') or not patient:
            raise ValueError('Invalid split or patient')
        if patient in patients and patients[patient] != split:
            raise ValueError(f'Patient leakage: {patient}')
        patients[patient] = split
        image = Path(row['image']).resolve()
        if image in seen or not image.is_file():
            raise ValueError(f'Duplicate or missing image: {image}')
        seen.add(image)
        if not row['boxes']:
            raise ValueError('Unannotated images are not assumed negative')
        for x1, y1, x2, y2 in row['boxes']:
            if not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                raise ValueError(f'Invalid bounding box: {image}')
    return data['records']


class CTDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        image = load_image(row['image'])
        if tuple(image.shape[-2:]) != (row['height'], row['width']):
            raise ValueError(f'PNG dimensions changed: {row["image"]}')
        boxes = torch.tensor(row['boxes'], dtype=torch.float32)
        return image, {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch))


def evaluate(model, loader, device, threshold):
    import torch
    from torchvision.ops import box_iou
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                keep = output['scores'] >= threshold
                boxes = output['boxes'][keep].cpu()
                scores = output['scores'][keep].cpu()
                truth = target['boxes']
                used = set()
                overlaps = box_iou(boxes, truth)
                for i in scores.argsort(descending=True).tolist():
                    candidates = [(float(overlaps[i, j]), j) for j in range(len(truth)) if j not in used]
                    best = max(candidates, default=(0.0, -1))
                    if best[0] >= 0.5:
                        tp += 1
                        used.add(best[1])
                    else:
                        fp += 1
                fn += len(truth) - len(used)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-12),
                tp=tp, fp=fp, fn=fn, score_threshold=threshold, iou_threshold=0.5)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    rows = read_manifest(args.manifest)
    subsets = [[row for row in rows if row['split'] == split] for split in ('train', 'val')]
    if not all(subsets):
        raise ValueError('Training requires nonempty train/val splits from different patients')
    loaders = [DataLoader(CTDataset(subset), batch_size=args.batch_size,
                          shuffle=i == 0, collate_fn=collate, num_workers=0) for i, subset in enumerate(subsets)]
    torch.manual_seed(42)
    model = build_model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    if args.model.exists():
        raise FileExistsError('Choose a new --model path to preserve the existing checkpoint')
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loaders[0]:
            losses = model([im.to(args.device) for im in images],
                           [{k: v.to(args.device) for k, v in t.items()} for t in targets])
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics = evaluate(model, loaders[1], args.device, args.threshold)
        print(json.dumps(dict(epoch=epoch + 1, loss=total / len(loaders[0]), validation=metrics)), flush=True)
        if metrics['f1'] > best:
            best = metrics['f1']
            torch.save(dict(format_version=1, architecture='alexnet_fasterrcnn',
                            model=model.state_dict(), epoch=epoch + 1, validation=metrics), args.model)


def predict(args):
    import torch
    from PIL import Image, ImageDraw
    model = build_model().to(args.device)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    if checkpoint.get('architecture') != 'alexnet_fasterrcnn':
        raise ValueError('Expected an AlexNet Faster R-CNN checkpoint')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        result = model([load_image(args.image).to(args.device)])[0]
    detections = [dict(box=box, score=score) for box, score in
                  zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist()) if score >= args.threshold]
    args.output.mkdir(parents=True, exist_ok=True)
    json_path, overlay_path = args.output / 'predictions.json', args.output / 'predictions.png'
    if json_path.exists() or overlay_path.exists():
        raise FileExistsError('Choose a new prediction output directory')
    with Image.open(args.image) as source:
        overlay = source.convert('RGB')
    draw = ImageDraw.Draw(overlay)
    for item in detections:
        draw.rectangle(item['box'], outline='red', width=2)
        draw.text(item['box'][:2], f'lesion {item["score"]:.2f}', fill='yellow')
    overlay.save(overlay_path)
    json_path.write_text(json.dumps(dict(image=str(args.image), detections=detections), indent=2), encoding='utf-8')
    print(f'{len(detections)} detections; output: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--model', type=Path, default=Path('artifacts/lesion_detector.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/prediction'))
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < float('inf') or not 0 <= args.threshold <= 1:
        parser.error('Invalid training parameters or threshold')
    if args.mode == 'train' and args.manifest is None:
        parser.error('--manifest is required')
    if args.mode == 'predict' and args.image is None:
        parser.error('--image is required')
    (train if args.mode == 'train' else predict)(args)


if __name__ == '__main__':
    main()
