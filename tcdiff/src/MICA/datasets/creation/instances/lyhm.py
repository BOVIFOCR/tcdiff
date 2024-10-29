
















from abc import ABC
from glob import glob
from pathlib import Path

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import RotateAxisAngle

from instances.instance import Instance


class LYHM(Instance, ABC):
    def __init__(self):
        super(LYHM, self).__init__()
        self.dst = '/MICA/OnFlame/LYHM/'
        self.src = '/MICA/LYHM/'


    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + '/*')):
            images[Path(actor).name] = glob(f'/{actor}/*.png')

        return images

    def get_flame_params(self):
        prams = {}
        for actor in sorted(glob(self.get_src() + '/*')):
            prams[Path(actor).name] = glob(f'/{actor}/*.npz')

        return prams

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + '/*')):
            all = glob(f'/{actor}/*.obj')
            registrations[Path(actor).name] = list(filter(lambda m: 'model_fit' not in m, all))

        return registrations

    def get_meshes(self):
        meshes = {}
        for actor in sorted(glob(self.get_src() + '/*')):
            meshes[Path(actor).name] = glob(f'/{actor}/scan/*.obj')

        return meshes

    def transform_mesh(self, path):
        mesh = load_objs_as_meshes(path, device=self.device)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]
        mesh.scale_verts_(0.01)

        rot = RotateAxisAngle(-45, axis='X', device=self.device)
        mesh._verts_list = [rot.transform_points(mesh.verts_list()[0])]
        rot = RotateAxisAngle(-45, axis='Y', device=self.device)
        mesh._verts_list = [rot.transform_points(mesh.verts_list()[0])]

        return mesh.clone()
