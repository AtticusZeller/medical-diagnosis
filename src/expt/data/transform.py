import torchvision.transforms.v2 as v2


def to_tensor_transform() -> v2.Compose:
    """
    Transform to convert image to tensor.

    Returns
    -------
    v2.Compose
        Composed transforms
    """
    return v2.Compose([v2.ToImage()])
