import cv2
import numpy as np

def normalize(vector):
    return np.asarray(vector) / np.linalg.norm(vector)

class Plane:
    def __init__( self, point, direction ):
        assert len(point), "Not a 3D point"
        assert len(direction), "Not a 3D direction"

        self.q = point
        self.n = normalize(direction)

class Line:
    def __init__( self, point, direction ):
        assert len(point), "Not a 3D point"
        assert len(direction), "Not a 3D direction"

        self.q = np.asarray(point)
        self.v = normalize(direction)

def plane_line_intersection( plane: Plane, line: Line ):
    """
    Find intersection of a plane and multiple lines.

    Parameters
    ----------
    plane : Plane
        plane containing a point and a normal vector.
    line : Line
        line containing a point and a direction vector.
    Returns 
    ----------
    3D coordinate of point intersecting a line and a plane.
    """
    planePoint = plane.q
    planeNormal = plane.n
    linePoint = line.q
    lineDirection = line.v

    if np.dot(lineDirection, planeNormal) < 1e-8:
        raise RuntimeError("Line and Plane do not intersect")

    return linePoint + lineDirection * np.dot(planeNormal, (planePoint - linePoint)) / np.dot(planeNormal, lineDirection)

def plane_lines_intersection( plane: Plane, lines: list[Line] ) -> np.ndarray:
    """
    Find intersection of a plane and multiple lines.

    Parameters
    ----------
    plane : Plane
        plane containing a point and a normal vector.
    lines : array_like
        list of N > 1 lines .
    Returns 
    ----------
    3D coordinate of closes point of multiple lines intersecting on a plane.
    """
    return np.average(np.array([plane_line_intersection(plane, line) for line in lines]), axis=0)

def intersect_lines( lines: list[Line] ) -> np.ndarray:
    """
    This equation for the formula to find intersection of many lines 
    is described [here](https://en.wikipedia.org/wiki/Line–line_intersection#In_three_dimensions_2).

    Parameters
    ----------
    lines : array_like
        list of N > 1 lines 
    Returns 
    ----------
    3D coordinate of closest point to all lines.
    """
    assert len(lines) > 1, "Need at least 2 lines to triangulate"
    As = [np.outer(line.v, line.v) - np.eye(3) for line in lines]
    Bs = [np.matmul(A, line.q).ravel() for A, line in zip(As, lines)]

    A = np.sum(np.stack(As, axis=2), axis=2)
    B = np.sum(np.stack(Bs, axis=1), axis=1)

    return np.linalg.inv(A) @ B

def point_line_distance(p: np.ndarray, line: Line):
    return np.linalg.norm(np.cross(line.v, p - line.q)) / line.v

def fit_line( points: list ) -> Line:
    """
    Fit line through list of points using
    Singular Value Decomposition.
    Parameters
    ----------
    points : array_like
        list of N points (all need to be the same dimension).
    Returns
    ----------
    Line
        line containing a point and the direction
    """
    points = np.asarray(points)
    C = np.mean(points, axis=0)
    _, _, V = np.linalg.svd(points - C)

    return Line(C, V[0])

def fit_plane( points: list ) -> Plane:
    """
    Fit plane through list of points using
    Singular Value Decomposition.
    Parameters
    ----------
    points : array_like
        list of N points (all need to be the same dimension).
    Returns
    ----------
    Plane
        plane containing a point and the normal
    """
    # Find the average of points (centroid) along the columns
    C = np.average(points, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = points - C
    # Singular value decomposition
    _, _, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    return Plane(C, N)

def undistort_camera_points(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(points2D.astype(np.float32).reshape((-1, 1, 2)), K, dist_coeffs).reshape((-1, 2))

def homogeneous_coordinates(
        points2D: np.ndarray
        ) -> np.ndarray:
    return np.concatenate([points2D, np.ones((points2D.shape[0], 1))], axis=1)

def camera_to_ray_world(
        points2D: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray) -> list[Line]:
    """
    Parameters
    ----------
    points2D : array_like
        list of N 2D pixel coordinates
    r : array_like
        3x1 camera orientation vector in world coordinate system
    t : array_like
        3x1 camera translation vector in world coordinate system
    K : array_like
        3x3 camera intrinsic matrix
    dist_coeffs : array_like 
        distortion coefficients of camera
    
    Returns
    ----------
    lines
        List of N lines in world coordinate system
    """
    r = np.asarray(r)
    if r.shape==(3,1) or r.shape==(1,3):
        r, _ = cv2.Rodrigues(r)
    xy = undistort_camera_points(points2D, K, dist_coeffs)
    directions = homogeneous_coordinates(xy)
    lines = [Line(t, np.matmul(r,direction)) for direction in directions]
    return lines

def camera_to_ray(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs=None) -> list[Line]:
    """
    Parameters
    ----------
    points2D : array_like
        list of N 2D pixel coordinates
    K : array_like
        3x3 camera intrinsic matrix
    dist_coeffs : array_like
        distortion coefficients of camera
    Returns
    ----------
    list
        List of N lines in camera coordinate system
    """
    return camera_to_ray_world(points2D, np.zeros(3), np.zeros(3), K, dist_coeffs)