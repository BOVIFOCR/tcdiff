
















from abc import ABC
from glob import glob
from pathlib import Path

from instances.instance import Instance


class FaceWarehouse(Instance, ABC):
    def __init__(self):
        super(FaceWarehouse, self).__init__()
        self.dst = '/MICA/OnFlame/FACEWAREHOUSE/'
        self.src = '/MICA/FACEWAREHOUSE/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'Images/*')):
            images[Path(actor).stem] = glob(f'{actor}/*.png')

        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_fits/*')):
            params[Path(actor).stem] = [sorted(glob(f'{actor}/*.npz'))[0]]

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_fits/*')):
            registrations[Path(actor).stem] = [f'{actor}/tmp/pose_0__def_trafo_fit.obj']

        return registrations
