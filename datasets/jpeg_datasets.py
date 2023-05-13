import tensorflow as tf
from pathlib import Path

def jpeg_img_loader(path):
    """
    Args:
        path: str, path to the image
    Returns:
        image: tf.Tensor, shape=(height, width, 3), dtype=tf.uint8
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def JpegDataset(
    image_dir,
    batch_size,
    shuffle=True,
    drop_remainder=True,
    resize=None,
    normalize=True,
    **kwargs):
    """
    Args:
        image_dir: str, path to the directory containing the images
        batch_size: int
        shuffle: bool
        drop_remainder: bool
        resize: tuple, (height, width)
        normalize: bool
    Returns:
        dataset: tf.data.Dataset
    """
    image_dir = Path(image_dir)
    image_paths = [str(path) for path in image_dir.glob('*.jpg')]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(jpeg_img_loader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if resize is not None:
        dataset = dataset.map(lambda x: tf.image.resize(x, resize))
    if normalize:
        dataset = dataset.map(lambda x: x / 255.)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset