class DataAugmentation:
    def __init__(self, transform, n_augmentation=2):
        self.transform = transform
        self.n_augmentation = n_augmentation

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_augmentation)]
