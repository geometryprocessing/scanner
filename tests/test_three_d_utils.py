import unittest
import random

from utils.three_d_utils import *

DOUBLE_EPS = 1.0e-14
DOUBLE_EPS_SQ = 1.0e-28
FLOAT_EPS = 1.0e-7
FLOAT_EPS_SQ = 1.0e-14

class TestIntersection(unittest.TestCase):

    def test_fit_line(self):
        self.assertEqual(1, 1, "1 is 1")

    def test_fit_plane(self):
        points = []
        for _ in range(1000):
            points.append( [random.uniform(-10,10), random.uniform(-10,10), 0] )

        _, n = ThreeDUtils.fit_plane(points)
        result = n
        expected = np.array([0,0,1]).reshape((1,3))
        self.assertAlmostEqual(np.linalg.norm(abs(result) - expected), 0.0,
                               msg='Plane normal should be exclusively in z direction')

    def test_lines_intersection(self):
        line1 = [[1,0,0], [-1,0,0]]
        line2 = [[0,1,0], [0,-1,0]]
        line3 = [[0,0,1], [0,0,-1]]

        result = ThreeDUtils.intersect_lines([line1,line2,line3])
        expected = np.array([0, 0, 0]).reshape((1,3))
        
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Lines should intersect at origin')
    
    def test_plane_line_intersection(self):
        plane = [[0,0,0], [0,0,1]]
        line = [[0,0,3], [1,1,-1]]
        result = ThreeDUtils.intersect_line_with_line(line[0], line[1], plane[0], plane[1])
        expected = np.array([3,3, 0]).reshape((1,3))

        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
                               msg='Line should intersect plane at [3,3,0]')

    def test_point_line_distance(self):
        point = [1,0,1]
        line = [[0,0,0], [0,0,1]]
        result = ThreeDUtils.point_line_distance(point, line)
        expected = 1.0

        self.assertAlmostEqual(result, expected,
                               msg='Point and line should have 1.0 (unitless) distance')


if __name__ == '__main__':
    unittest.main()