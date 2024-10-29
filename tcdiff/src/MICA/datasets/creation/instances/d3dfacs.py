
















from abc import ABC
from glob import glob
from pathlib import Path

from instances.instance import Instance


class D3DFACS(Instance, ABC):
    def __init__(self):
        super(D3DFACS, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/D3DFACS/'
        self.src = '/home/wzielonka/datasets/D3DFACS/'

    def get_images(self):
        images = {}
        for file in sorted(glob(self.get_src() + 'processed/images/*')):
            actor = Path(file).stem
            images[actor] = glob(f'{file}/*.jpg')

        return images

    def get_flame_params(self):
        params = {}
        for file in sorted(glob(self.get_src() + 'processed/FLAME/*.npz')):
            actor = Path(file).stem
            params[actor] = [file]

        return params

    def get_registrations(self):
        registrations = {}
        for file in sorted(glob(self.get_src() + 'processed/registrations/*')):
            actor = Path(file).stem.split('_')[0]
            registrations[actor] = [file]

        return registrations
