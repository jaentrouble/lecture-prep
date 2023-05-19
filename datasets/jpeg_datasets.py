import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
import tqdm

class JpegDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            resize: tuple = None,
            normalize: bool = True,
        ) -> None:
        """
        Args:
            image_dir: str, path to the directory containing the images
            resize: tuple, (height, width)
            normalize: bool or str
                True: normalize to [0,1]
                'tanh': normalize to [-1,1]
        """
        super().__init__()
        self.image_dir = Path(image_dir)
        self.image_paths = [str(path) for path in self.image_dir.glob('*.jpg')]
        self.image_paths.sort()
        self.images = []
        for path in tqdm.tqdm(
                self.image_paths,
                desc='Loading images',
                leave=False,
            ):
            image = read_image(path)
            if normalize:
                if normalize == 'tanh':
                    image = image / 127.5 - 1.
                else:
                    image = image / 255.
            if resize is not None:
                image = resize(image, resize)
            self.images.append(image)

        self.n = len(self.images)

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]