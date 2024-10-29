
















from abc import ABC
from glob import glob
from pathlib import Path

import numpy as np

from instances.instance import Instance


class Florence(Instance, ABC):
    def __init__(self):
        super(Florence, self).__init__()
        self.dst = '/MICA/OnFlame/FLORENCE/'
        self.src = '/MICA/FLORENCE/'

    def get_min_det_score(self):

        return 0.55

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            imgs = sorted(glob(f'{actor}/*.jpg'))
            
            indecies = np.arange(len(imgs))
            
            images[Path(actor).stem] = [imgs[i] for i in indecies]
            
        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/iter1/*')):
            params[Path(actor).stem] = glob(f'{actor}/*.npz')

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/iter1/*')):
            if 'rendering' in actor:
                continue
            registrations[Path(actor).stem] = glob(f'{actor}/*.obj')

        return registrations
