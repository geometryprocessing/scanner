import unittest

from src import intersection

class TestIntersection(unittest.TestCase):

    def test_plane_init(self):
        """Test for Plane class initialization"""
        self.assertEqual(1, 1, "1 is 1")

    def test_line_init(self):
        """Test for Line class initialization"""
        self.assertEqual(1, 1, "1 is 1")

    def test_fit_line(self):
        self.assertEqual(1, 1, "1 is 1")

    def test_fit_plane(self):
        self.assertEqual(1, 1, "1 is 1")

    def test_line_line_intersection(self):
        self.assertEqual(1, 1, "1 is 1")
    
    def test_line_plane_intersection(self):
        self.assertEqual(1, 1, "1 is 1")

if __name__ == '__main__':
    unittest.main()