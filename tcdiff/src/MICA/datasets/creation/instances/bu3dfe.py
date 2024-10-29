
















from abc import ABC
from glob import glob
from pathlib import Path

from pytorch3d.io import load_objs_as_meshes

from instances.instance import Instance


class BU3DFE(Instance, ABC):
    def __init__(self):
        super(BU3DFE, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/BU3DFE/'
        self.src = '/scratch/NFC/BU-3DFE/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src().replace('BU-3DFE', 'BU-3DFE_clean') + 'images/*')):
            images[Path(actor).name] = glob(f'{actor}/*.jpg')

        return images

    def get_flame_params(self):
        prams = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/iter2/*')):
            prams[Path(actor).name] = glob(f'{actor}/*.npz')

        return prams

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/iter2/neutral_align/*')):
            registrations[Path(actor).name] = glob(f'{actor}/*.obj')

        return registrations

    def get_meshes(self):
        meshes = {}
        files = sorted(glob(self.get_src() + 'raw_ne_data/*'))
        actors = set(map(lambda f: Path(f).name[0:5], files))
        for actor in actors:
            meshes[Path(actor).name] = next(filter(lambda f: actor in f and 'obj' in f, files))

        return meshes

    def transform_mesh(self, path):
        self.update_obj(path)
        mesh = load_objs_as_meshes([path], device=self.device)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]
        mesh.scale_verts_(0.01)

        return mesh.clone()
