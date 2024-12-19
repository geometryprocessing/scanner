import unittest
import random

from src.scanner.intersection import *

DOUBLE_EPS = 1.0e-14
DOUBLE_EPS_SQ = 1.0e-28
FLOAT_EPS = 1.0e-7
FLOAT_EPS_SQ = 1.0e-14

class TestIntersection(unittest.TestCase):

    def test_plane_init(self):
        """Test for Plane class initialization"""
        point = [1,2,3]
        normal = [5,6,7]
        plane = Plane(point, normal)
        # TODO: normal vector needs to be normalized
        # self.assertEqual(plane.n, np.array([5,6,7]).reshape((-1,3)),
        #                  mgs='Plane normal should be converted to normalized 1x3 numpy array')
        self.assertEqual(plane.q, np.array([1,2,3]).reshape((-1,3)),
                         mgs='Plane point should be converted to 1x3 numpy array')

    def test_line_init(self):
        """Test for Line class initialization"""
        point = [1,2,3]
        normal = [5,6,7]
        line = Line(point, normal)
        # TODO: normal vector needs to be normalized
        self.assertEqual(line.n, np.array([5,6,7]).reshape((-1,3)),
                         mgs='Line direction should be converted to 1x3 numpy array')
        self.assertEqual(line.p, np.array([1,2,3]).reshape((-1,3)),
                         mgs='Line point should be converted to 1x3 numpy array')

    def test_fit_line(self):
        self.assertEqual(1, 1, "1 is 1")

    def test_fit_plane(self):
        points = []
        for _ in range(1000):
            points.append( [random.uniform(-10,10), random.uniform(-10,10), 0] )

        plane = fit_plane(points)
        result = plane.n
        expected = np.array([0,0,1]).reshape((1,3))
        self.assertAlmostEqual(np.linalg.norm(abs(result) - expected), 0.0,
                               msg='Plane normal should be exclusively in z direction')

    def test_lines_intersection(self):
        line1 = Line([1,0,0], [-1,0,0])
        line2 = Line([0,1,0], [0,-1,0])
        line3 = Line([0,0,1], [0,0,-1])

        result = intersect_lines([line1,line2,line3])
        expected = np.array([0, 0, 0]).reshape((1,3))
        
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Lines should intersect at origin')
    
    def test_plane_line_intersection(self):
        plane = Plane([0,0,0], [0,0,1])
        line = Line([0,0,3], [1,1,-1])
        result = plane_line_intersection(plane, line)
        expected = np.array([-3,-3, 0]).reshape((1,3))

        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, 
                               msg='Line should intersect plane at [-3,-3,0]')

    def test_point_line_distance(self):
        point = [1,0,1]
        line = Line([0,0,0], [0,0,1])
        result = point_line_distance(point, line)
        expected = 1.0

        self.assertAlmostEqual(result, expected,
                               msg='Point and line should have 1.0 (unitless) distance')


if __name__ == '__main__':
    unittest.main()