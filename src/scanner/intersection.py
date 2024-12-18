import cv2
import numpy as np

def normalize(vector) -> np.ndarray:
    """
    Normalizes input vector. If vector is all zero, returns original vector.

    Parameters
    ----------
    vector : array_like
        vector of shape Nx1 or 1xN

    Returns
    ----------
    Normalized vector of original shape.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return np.array(vector) / np.linalg.norm(vector)

class Plane:
    def __init__( self, point, direction ):
        self.q = np.array(point).reshape((1,-1))
        self.n = normalize(direction).reshape((1,-1))

class Line:
    def __init__( self, point, direction ):
        self.q = np.array(point).reshape((1,-1))
        self.v = normalize(direction).reshape((1,-1))

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
    planeNormalTranspose = planeNormal.T
    linePoint = line.q
    lineDirection = line.v

    if abs(np.dot(lineDirection, planeNormalTranspose)) < 1e-8:
        raise RuntimeError("Line and Plane do not intersect")

    return linePoint + lineDirection * np.dot(planeNormalTranspose, (planePoint - linePoint)) / np.dot(planeNormalTranspose, lineDirection)

def plane_lines_intersection( plane: Plane, lines: np.ndarray[Line] ) -> np.ndarray:
    """
    TODO: vectorize operations to avoid using list comprehension.
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

def intersect_lines( lines: np.ndarray[Line] ) -> np.ndarray:
    """
    TODO: vectorize operations to avoid using list comprehension.
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
    p = np.array(p).reshape((1,-1))
    return np.linalg.norm(np.cross(line.v, p - line.q)) / np.linalg.norm(line.v)

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
    points = np.array(points)
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
        dist_coeffs: np.ndarray,
        R: np.ndarray=None,
        P: np.ndarray=None ) -> np.ndarray:
    """
    TODO: understand better the functionality of cv2.undistortPoints
    In order to find where the points would be from undistortion, pass the same matrix K for P.
    """
    return cv2.undistortPoints(np.array(points2D, dtype=np.float32).reshape((-1, 1, 2)), K, dist_coeffs, R, P).reshape((-1, 2))

def homogeneous_coordinates(
        points2D: np.ndarray
        ) -> np.ndarray:
    """
    Converts coordinates into homogenous form, i.e.
    the points (x, y) becomes (x, y, 1)

    Parameters
    ----------
    points2D : array_like
        list of N 2D coordinates (shape Nx2)
    
    Returns
    ----------
    lines
        List of N 2D homogenous coordinates (shape Nx3).
    """
    return np.concatenate([points2D, np.ones((points2D.shape[0], 1))], axis=1)

def camera_to_ray_world(
        points2D: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray) -> np.ndarray[Line]:
    """
    Converts camera pixel coordinates in x,y to rays into the world.
    It additionally performs rotation and translation so that rays
    are in world coordinates.

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
    r = np.array(r)
    if r.shape==(3,1) or r.shape==(1,3):
        r, _ = cv2.Rodrigues(r)
    xy = undistort_camera_points(points2D, K, dist_coeffs)
    directions = homogeneous_coordinates(xy)
    lines = [Line(t, np.matmul(r,direction)) for direction in directions]
    return lines

def camera_to_ray(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray=None) -> np.ndarray[Line]:
    """
    Function provided for convenience.
    It differs from camera_to_ray_world() only in what argument(s) it accepts.

    Converts camera pixel coordinates in x,y to rays into the world.
    It does not perform rotation or translation, therefore it keeps the rays
    in camera coordinates where camera is origin.

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

def combine_transformations(R1: np.ndarray,
                            T1: np.ndarray,
                            R2: np.ndarray,
                            T2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    R1 : array_like
        First rotation matrix (3x3)
    T1 : array_like
        First translation vector (1x3 or 3x1)
    R2 : array_like
        Second rotation matrix (3x3)
    T2 : array_like
        Second translation vector (1x3 or 3x1)

    Returns
    ----------
    R
        Combined rotation matrix (3x3)
    T
        Combined translation vector (3x1)
    """
    R1 = np.array(R1).reshape((3,3))
    T1 = np.array(T1).reshape((3,1))
    R2 = np.array(R2).reshape((3,3))
    T2 = np.array(T2).reshape((3,1))

    R = np.dot(R1, R2)
    T = np.dot(R2, T1) + T2

    return R, T