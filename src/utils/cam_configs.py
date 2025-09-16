import inspect
import sys
import numpy as np

class CameraConfig:
    def __init__(self):
        self.pretty_name = 'Toy Camera'
        self.name = 'toycamera'
        self.R = np.asarray([0.,0.,0.])
        self.T = np.asarray([0.,0.,0.])
        self.K = np.asarray([[],
                             [],
                             [0., 0., 1.]])
        self.dist_coeffs = None

class ArenaCamera(CameraConfig):
    def __init__(self):
        super.__init__()
        self.pretty_name = 'Arena 32Mpx Camera'
        self.name = 'arena'
        self.width = 6464
        self.height = 4852
        self.K = np.asarray([[1.4852e04, 0, 3.1818e03], 
                             [0, 1.48677e04, 2.46895e03],
                             [0,0,1]])
        self.dist_coeffs = np.asarray([-1.0079e-01,
                                       -9.6801e-01,
                                       4.155e-04,
                                       -4.9249e-04,
                                       8.8487e00])

class ChronosCamera(CameraConfig):
    def __init__(self):
        super.__init__()
        self.pretty_name = 'Chronos HD Camera'
        self.name = 'chronos'
        self.width = 1920
        self.height = 1080
        self.K = np.asarray([[3580.915139167154,0.0,967.7411991347725],
                            [0.0,3578.1633141223797,509.395929381727],
                            [0.0,0.0,1.0]])
        self.dist_coeffs = np.asarray([-0.05202839890657609,
                                       -0.05977211653771139,
                                       0.0014034145040185428,
                                       -0.00031193552659360146,
                                       0.3585615042719389])

class TritonCamera(CameraConfig):
    def __init__(self):
        super.__init__()
        self.width = 2448
        self.height = 2048
        self.pretty_name = 'Triton10 5Mpx Camera'
        self.name = 'triton'

class Triton1Camera(TritonCamera):
    def __init__(self):
        super.__init__()
        self.pretty_name = 'Triton10 5Mpx Camera No.1'
        self.name = 'triton1'
        self.K = np.asarray([[4398.865805444324, 0.0, 1224.998494272802],
                             [0.0,4395.209485484965,1010.5491943764957],
                             [0.0,0.0,1.0]])
        self.dist_coeffs = np.asarray([-0.18346018257897984,
                                        0.31643440753823476,
                                        0.0003134068085886509,
                                        0.00020178521826665733,
                                        0.11827598690686253])
        
class Triton2Camera(TritonCamera):
    def __init__(self):
        super.__init__()
        self.pretty_name = 'Triton10 5Mpx Camera No.2'
        self.name = 'triton2'
        self.K = np.asarray([[4415.772826208225,0.0,1214.1772432727691],
                             [0.0,4412.163698025551,1041.2152001109412],
                             [0.0,0.0,1.0]])
        self.dist_coeffs = np.asarray([-0.1845207964010922,
                                        0.3764315875260389,
                                        0.0006885791954024719,
                                        0.00010983578916095759,
                                       -0.04955677781648061])
        self.R = np.asarray([   0.02470863, -2.22986647,    0.10327809])
        self.T = np.asarray([ 562.73912308, 30.61416842, 1021.49771131])