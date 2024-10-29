
















from abc import ABC
from glob import glob
from pathlib import Path

import numpy as np
from pytorch3d.io import load_objs_as_meshes

from instances.instance import Instance


class PB4D(Instance, ABC):
    def __init__(self):
        super(PB4D, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/BP4D/'
        self.src = '/scratch/NFC/BP4D/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            imgs = sorted(glob(f'/{actor}/*.jpg'))
            indecies = np.random.choice(len(imgs), 100, replace=False)
            images[Path(actor).name] = [imgs[i] for i in indecies]

        return images

    def get_flame_params(self):
        prams = {}
        for file in sorted(glob(self.get_src() + 'FLAME_parameters/*.npz')):
            prams[Path(file).stem] = [file]

        return prams

    def get_registrations(self):
        registrations = {}
        for file in sorted(glob(self.get_src() + 'registrations/*')):
            registrations[Path(file).stem] = [file]

        return registrations

    def get_meshes(self):
        meshes = {}
        for file in sorted(glob(self.get_src() + 'scans/*.obj')):
            meshes[Path(file).stem] = [file]

        return meshes

    def transform_mesh(self, path):
        mesh = load_objs_as_meshes(path, device=self.device)
        mesh.scale_verts_(0.01)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]

        return mesh.clone()
