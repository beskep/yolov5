import json
from pathlib import Path
from shutil import copy2
from typing import Literal

import click
import numpy as np
from loguru import logger
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _imgread(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img


def _imgsize(path):
    # (width, height)
    return _imgread(path).size


def _fmt(x):
    return format(x, '.6f')


class Labelme2Yolov5:
    ENCODING = 'utf-8'
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    CLASSES: tuple
    CLS_ALL: set

    def __init__(self, src, dst='datasets/datasets') -> None:
        self.src = Path(src)
        self.dst = Path(dst)

        self.src.stat()

        try:
            self.dst.stat()
        except FileNotFoundError:
            self.dst.parent.stat()
            self.dst.mkdir()

    def _images(self):
        return sorted(x for x in self.src.rglob('*') if x.suffix in self.IMG_EXTS)

    def _mkdir(self):
        for d in ['images', 'labels']:
            p = self.dst / d
            p.joinpath('train').mkdir(parents=True, exist_ok=True)
            p.joinpath('test').mkdir(parents=True, exist_ok=True)

    def read_label(self, src: Path):
        path = src.with_suffix('.json')
        with path.open('r', encoding=self.ENCODING) as f:
            label = json.load(f)

        return label

    def classes(self):
        classes = set()
        for image in self._images():
            label = self.read_label(image)
            classes |= {x['label'] for x in label['shapes']}

        return classes

    def check_images(self):
        images = self._images()
        logger.info('images count: {}', len(images))

        for image in tqdm(images, desc='Checking...'):
            label = self.read_label(image)
            classes = {x['label'] for x in label['shapes']}
            if (s := classes - self.CLS_ALL):
                raise ValueError(f'Invalid classes {s} in "{image}"')

    def label(self, label: str) -> str:
        raise NotImplementedError

    def _shape_label(self, shape: dict, imgsize: tuple[int]):
        # TODO test

        if shape['shape_type'] != 'rectangle':
            raise ValueError("shape['shape_type'] != 'rectangle'")

        # [[x1, y1],
        #  [x2, y2]]
        apoints = np.array(shape['points'])

        rpoints = apoints / np.array(imgsize)
        rpoints[rpoints < 0] = 0.0
        rpoints[rpoints > 1] = 1.0

        cls = self.CLASSES.index(self.label(shape['label']))
        center = np.average(rpoints, axis=0)
        length = np.abs(rpoints[0] - rpoints[1])

        # class x_center y_center width height
        return [
            cls,
            _fmt(center[0]),
            _fmt(center[1]),
            _fmt(length[0]),
            _fmt(length[1]),]

    def _image_label(self, src: Path):
        label = self.read_label(src)
        imgsize = _imgsize(src)

        return [self._shape_label(s, imgsize) for s in label['shapes']]

    def _convert(self, images: list[Path], dataset: Literal['train', 'test']):
        for idx, image in enumerate(tqdm(images, desc=f'{dataset.title()} Set')):
            fname = f'{idx:06d}'
            labels = self._image_label(image)
            lp = self.dst / 'labels' / dataset / f'{fname}.txt'
            with lp.open('w', encoding=self.ENCODING) as f:
                for label in labels:
                    f.write(' '.join(str(x) for x in label) + '\n')

            copy2(image, self.dst / 'images' / dataset / f'{fname}{image.suffix}')

    def convert(self, test_size=0.2, train_size=None, random_state=None):
        images = self._images()
        if not images:
            raise FileNotFoundError(f'Images not found in "{self.src}"')

        train, test = train_test_split(
            images,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

        self._mkdir()
        self._convert(train, 'train')
        self._convert(test, 'test')


class Labelme2Equipment(Labelme2Yolov5):
    CLASSES = ('ACCU', 'FAN', 'PV')
    CLS_CNVT = {'AHU': 'ACCU'}
    CLS_ALL = set(CLASSES) | set(CLS_CNVT.keys())

    def __init__(self, src, dst='datasets/equipment') -> None:
        super().__init__(src, dst)

    def label(self, label: str):
        return self.CLS_CNVT.get(label, label)


class Labelme2Light(Labelme2Equipment):
    CLS_ALL = {
        'CFL', 'Diffuser', 'EHP-1way', 'EHP-4way', 'Fan_Heater', 'FAN', 'FCU-2way', 'FPL', 'LED_Circle', 'LED_DW',
        'LED_Rec', 'LED_Str', 'LED', 'PAC', 'RAC'}
    CLASSES = ('LED', 'FPL', 'etc')

    def __init__(self, src, dst='datasets/light') -> None:
        super().__init__(src, dst)

    def label(self, label: str):
        if label == 'FPL':
            return label

        if label.startswith('LED'):
            return 'LED'

        return 'etc'


@click.command()
@click.argument('src')
@click.argument('dst', required=False)
@click.option('-d', '--dataset', type=click.Choice(['equipment', 'light']), required=True)
@click.option('-t', '--train', type=int, required=True, help='train size')
@click.option('-v', '--val', type=int, required=True, help='validation size')
@click.option('-s', '--seed', type=int, help='random state seed')
@click.option('--check/--no-check', default=True, help='check dataset labels')
def main(src, dst, dataset, train, val, seed, check):
    """    e.g.

    \b python labelme2yolov5.py --dataset equipment --train 40 --test 10 "source/path"

    \b python labelme2yolov5.py -d light -t 40 -v 10 -s 42 --no-check "source/path" "destination/path"
    """
    dst = dst or f'datasets/{dataset}'

    logger.info('SRC: "{}"', src)
    logger.info('DST: "{}"', dst)
    logger.info('dataset: {} | train: {} | val: {} | seed: {} | check: {}', dataset, train, val, seed, check)

    if dataset == 'equipment':
        l2d = Labelme2Equipment(src=src, dst=dst)
    else:
        l2d = Labelme2Light(src=src, dst=dst)

    if check:
        l2d.check_images()

    with logger.catch():
        l2d.convert(test_size=val, train_size=train, random_state=seed)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
