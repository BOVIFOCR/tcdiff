
















import numpy as np
import torch

from generator import Generator
from instances.bu3dfe import BU3DFE
from instances.d3dfacs import D3DFACS
from instances.facewarehouse import FaceWarehouse
from instances.florence import Florence
from instances.frgc import FRGC
from instances.lyhm import LYHM
from instances.pb4d import PB4D
from instances.stirling import Stirling



np.random.seed(42)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    datasets = [FaceWarehouse(), LYHM(), FRGC(), Florence(), Stirling()]
    generator = Generator([Florence()])


    generator.run()
