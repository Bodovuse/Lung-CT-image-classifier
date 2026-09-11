"""Learn CT features using training labels; select weights on validation macro F1."""
import json
import math

from alexNet.alexNet import build_backbone, load_image


def validate_splits(rows):
    classes = ['lesion_absent', 'lesion_present']
    patients = {}
    for row in rows:
        previous = patients.setdefault(row['patient_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Patient leakage')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(classes):
            raise ValueError(f'{split} must contain both lesion classes')
    return classes


def fine_tune(rows, output, device='cpu', batch_size=16, epochs=20,
              patience=5, learning_rate=1e-5, seed=42):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.models import AlexNet_Weights
    from sklearn.metrics import f1_score

    classes = validate_splits(rows)
    if epochs < 1 or patience < 1 or batch_size < 1 or not 0 < learning_rate < math.inf:
        raise ValueError('Invalid fine-tuning parameters')
    torch.manual_seed(seed)
    transform = AlexNet_Weights.IMAGENET1K_V1.transforms()

    class Slices:
        def __init__(self, split):
            self.rows = [r for r in rows if r['split'] == split]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return transform(load_image(row['image'])), classes.index(row['label'])

    train_data, val_data = Slices('train'), Slices('val')
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    backbone = build_backbone(AlexNet_Weights.IMAGENET1K_V1).to(device)

    # A temporary differentiable head supplies gradients to the CNN.
    model = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
                          nn.Linear(256 * 6 * 6, 2)).to(device)
    counts = [sum(r['label'] == c for r in train_data.rows) for c in classes]
    weights = torch.tensor([len(train_data) / (2 * n) for n in counts], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best, stale, history = -1.0, 0, []
    checkpoint = output / 'alexnet_features.pt'

    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for step, (images, labels) in enumerate(train, 1):
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite fine-tuning loss')
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % 100 == 0 or step == len(train):
                print(f'AlexNet epoch {epoch}/{epochs}, batch {step}/{len(train)}, loss={total / step:.5f}', flush=True)
        model.eval()
        truth, predicted = [], []

        with torch.inference_mode():
            for images, labels in val:
                truth.extend(labels.tolist())
                predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
        score = float(f1_score(truth, predicted, labels=[0, 1], average='macro', zero_division=0))
        history.append(dict(epoch=epoch, training_loss=total / len(train), validation_macro_f1=score))
        (output / 'finetuning_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        
        print(f'AlexNet epoch {epoch}: validation macro F1={score:.5f}', flush=True)
        if score > best:
            best, stale = score, 0
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(dict(architecture='alexnet_ct_features', backbone=backbone.state_dict(),
                            auxiliary_head=model[-1].state_dict(), classes=classes, epoch=epoch,
                            validation_macro_f1=score, seed=seed,
                            preprocessing='AlexNet_Weights.IMAGENET1K_V1.transforms()'), temporary)
            temporary.replace(checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
        
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    backbone.load_state_dict(saved['backbone'])
    return backbone.eval().requires_grad_(False), dict(
        best_epoch=saved['epoch'], validation_macro_f1=saved['validation_macro_f1'],
        completed_epochs=len(history), learning_rate=learning_rate, patience=patience,
        selection_metric='validation_macro_f1', supervision='annotation-derived training labels')
