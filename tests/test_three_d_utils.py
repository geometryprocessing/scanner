import numpy
import numpy.random as npr
import numpy.testing as nptest
import unittest
import random

from src.utils.three_d_utils import *

DOUBLE_EPS = 1.0e-14
DOUBLE_EPS_SQ = 1.0e-28
FLOAT_EPS = 1.0e-7
FLOAT_EPS_SQ = 1.0e-14

class TestThreeDUtils(unittest.TestCase):

    def test_fit_line(self):
        _, v = fit_line([[0,0,1], [0,0,2], [0,0,3], [0,0,4]])
        result = v
        expected = np.array([0,0,1]).reshape((1,3))
        self.assertAlmostEqual(np.linalg.norm(abs(result) - expected), 0.0,
                               msg='Line direction should be exclusively in z direction')

    def test_normalize(self):
        result = normalize([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        expected = np.full(shape=(3,4), fill_value=0.5)
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='All normals should be an array filled with 0.5')

    def test_fit_plane(self):
        points = []
        for _ in range(1000):
            points.append( [random.uniform(-10,10), random.uniform(-10,10), 0] )

        _, n = fit_plane(points)
        result = n
        expected = np.array([0,0,1]).reshape((1,3))
        self.assertAlmostEqual(np.linalg.norm(abs(result) - expected), 0.0,
                               msg='Plane normal should be exclusively in z direction')

    def test_lines_intersection(self):
        # TODO: fix the intersect_lines function, since it's trying to normalize a (3,) array
        line1 = [[1,0,0], [-1,0,0]]
        line2 = [[0,1,0], [0,-1,0]]
        line3 = [[0,0,1], [0,0,-1]]

        result = intersect_lines([line1,line2,line3])
        expected = np.array([0, 0, 0]).reshape((1,3))
        
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Lines should intersect at origin')
    
    def test_plane_line_intersection(self):
        plane = [[0,0,0], [0,0,1]]
        line = [[0,0,3], [1,1,-1]]
        result = intersect_line_with_plane(line[0], line[1], plane[0], plane[1])
        expected = np.array([3,3, 0]).reshape((1,3))

        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
                               msg='Line should intersect plane at [3,3,0]')

    def test_point_line_distance(self):
        point = [1,0,1]
        line = [[0,0,0], [0,0,1]]
        result = point_line_distance(point, line[0], line[1])
        expected = 1.0

        self.assertAlmostEqual(result, expected,
                               msg='Point and line should have distance of 1.0')
        
    # def test_get_origin(self):
    #     R = 
    #     T = 
    #     result = ThreeDUtils.get_origin(R,T)
    #     expected = np.array([[],[],[]])
    #     self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
    #                            msg='Point and line should have distance of 1.0')
        
    # def test_rotation_combine_transformations(self):
    #     R1 = np.eye(3)
    #     T1 = np.zeros(shape=(3,1))
    #     R2 = np.array([[0,0,1], [0,1,0], [1,0,0]])
    #     T2 = np.array([1,1,1])
    #     result, _ = ThreeDUtils.combine_transformations(R1,T1,R2,T2)
    #     expected = np.array([[], [], []])
    #     self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
    #                            msg='Point and line should have distance of 1.0')
        
    # def test_translation_combine_transformations(self):
    #     R1 = np.eye(3)
    #     T1 = np.zeros(shape=(3,1))
    #     R2 = np.array([[0,0,1], [0,1,0], [1,0,0]])
    #     T2 = np.array([1,1,1])
    #     _, result = ThreeDUtils.combine_transformations(R1,T1,R2,T2)
    #     expected = np.array([[], [], []])
    #     self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
    #                            msg='Point and line should have distance of 1.0')

if __name__ == '__main__':
    unittest.main()