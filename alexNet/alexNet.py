"""Shared AlexNet backbone and image loading for SVM and Random Forest classification."""

def build_backbone(weights=None):
    from torchvision.models import alexnet
    return alexnet(weights=weights).features[:-1]


def load_image(path):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode not in ('L', 'RGB'):
            raise ValueError(f'Expected an 8-bit grayscale or RGB PNG: {path}')
        return to_tensor(image.convert('RGB'))


if __name__ == '__main__':
    # Support both python -m alexNet.alexNet and the original script path.
    import sys
    from pathlib import Path
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alexNet.compare_classifiers import main
    main()
