import os.path
from os import environ, makedirs, listdir, rename
from os.path import exists, expanduser, join, splitext, basename
from typing import Optional, AnyStr, cast
from urllib.request import urlretrieve


def get_data_dir(data_dir: Optional[AnyStr] = None) -> AnyStr:
    if data_dir is None:
        data_dir = cast(AnyStr, environ.get('SLDL', join('~', 'sldl_data')))
        data_dir = expanduser(data_dir)

    if not exists(data_dir):
        makedirs(data_dir)

    return data_dir


def get_checkpoint_path(url):
    name = url.split('/')[-1]
    data_root = get_data_dir()
    full_data_path = join(data_root, name)

    if not exists(full_data_path):
         urlretrieve(url, full_data_path)
    
    return full_data_path
