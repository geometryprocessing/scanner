import cv2
import numpy as np
import open3d as o3d

from utils.image_utils import ImageUtils

def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes one-dimensional vector. If vector is all zero, returns original vector.

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
    return np.array(vector, dtype=np.float32) / norm

class Plane:
    def __init__( self, point, direction ):
        """
        Plane class

        Parameters
        ----------
        point : array_like
            point present on plane. Class saves it as numpy array and with shape 1xN.
        direction : array_like
            normal direction of plane. Class saves it as a normalized numpy array and with shape 1xN.
        """
        self.q = np.array(point, dtype=np.float32).reshape((1,-1))
        self.n = normalize(direction).reshape((1,-1))

class Line:
    def __init__( self, point, direction ):
        """
        Line class

        Parameters
        ----------
        point : array_like
            point present on plane. Class saves it as numpy array and with shape 1xN.
        direction : array_like
            direction of line. Class saves it as a normalized numpy array and with shape 1xN.
        """
        self.q = np.array(point, dtype=np.float32).reshape((1,-1))
        self.v = normalize(direction).reshape((1,-1))

def fit_line( points: list ) -> Line:
    """
    Fit line through list of points using
    Singular Value Decomposition.

    Parameters
    ----------
    points : array_like
        list of N points (all points need to have the same number of dimensions).

    Returns
    ----------
    Line
        line containing a point and the direction that best fit the list points
    """
    points = np.array(points, dtype=np.float32)
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
        list of N points (all points need to have the same number of dimensions).

    Returns
    ----------
    Plane
        plane containing a point and the normal that best fit the list of points
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

class ThreeDUtils:
    @staticmethod
    def plane_line_intersection( plane: Plane, line: Line ):
        """
        Find intersection of a plane and multiple lines.

        Parameters
        ----------
        plane : Plane
            plane class containing a point and a normal vector.
        line : Line
            line class containing a point and a direction vector.

        Returns 
        ----------
        numpy array of shape 1xN of point where line and plane intersect.
        """
        planePoint = plane.q
        planeNormalTranspose = plane.n.T
        linePoint = line.q
        lineDirection = line.v

        if abs(np.dot(lineDirection, planeNormalTranspose)) < 1e-8:
            raise RuntimeError("Line and Plane do not intersect")

        return linePoint + lineDirection * np.dot((planePoint - linePoint), planeNormalTranspose) / np.dot(lineDirection, planeNormalTranspose)

    @staticmethod
    def plane_lines_intersection( plane: Plane, lines: np.ndarray[Line] ) -> np.ndarray:
        """
        TODO: vectorize operations to avoid using list comprehension.
        Find intersection of a plane and multiple lines.

        Parameters
        ----------
        plane : Plane
            plane class containing a point and a normal vector.
        lines : array_like
            list of N > 1 line classes, each containing a point and a direction vector.

        Returns 
        ----------
        numpy array of closest point of multiple lines intersecting on a plane.
        """
        return np.average(np.array([ThreeDUtils.plane_line_intersection(plane, line) for line in lines], dtype=np.float32), axis=0)

    @staticmethod
    def find_lambda(origin1: np.ndarray,
                    directions1: np.ndarray,
                    origin2: np.ndarray,
                    directions2: np.ndarray) -> np.ndarray:
        """
        Finds the length/depth (lambda) along the camera 1 rays

        Parameters
        ----------
        origin1 : array_like
            origin (shape 1x3) of camera 1
        directions1 : array_like
            camera 1 rays (shape Nx3)
        origin2 : array_like
            origin (shape 1x3) of camera 2
        directions2 : array_like
            camera 2 rays (shape Nx3)

        Returns 
        ----------
        L
            numpy array of lambdas (shape Nx1), where lambda is the depth along
            camera 1 rays, i.e. p_w = origin1 + directions1 * L, where p_w
            are the world coordinates (shape Nx3)
        """
        v12 = np.sum(np.multiply(directions1, directions2), axis=1).reshape(-1,1)
        v1 = np.linalg.norm(directions1, axis=1).reshape(-1,1)**2
        v2 = np.linalg.norm(directions2, axis=1).reshape(-1,1)**2

        L = (np.matmul(directions1, origin2.T - origin1.T) * v1 + np.matmul(directions2, origin1.T - origin2.T) * v12) / (v1 * v2 - v12**2)
        return L

    @staticmethod
    def triangulate( line1: Line, line2: Line ) -> np.ndarray:
        """
        TODO: discard function

        Parameters
        ----------
        line1 : Line
            line class containing a point and a direction vector.
        line2 : Line
            line class containing a point and a direction vector.

        Returns 
        ----------
        triangulated point of two lines.
        """
        T1 = line1.q
        T2 = line2.q
        dir1 = line1.v
        dir2 = line2.v
        
        v12 = np.sum(np.multiply(dir1, dir2), axis=1)
        v1 = np.linalg.norm(dir1, axis=1)
        v2 = np.linalg.norm(dir2, axis=1)

        lambda1 = (np.matmul(dir1, T2 - T1) * v1**2 + np.matmul(dir1, T1 - T2) * v12)   / (v1**2 * v2**2 - v12**2)
        lambda2 = (np.matmul(dir1, T2 - T1) * v12.T + np.matmul(dir2, T1 - T2) * v2**2) / (v1**2 * v2**2 - v12**2)

        # find the midpoint between the two lines for every point
        return ( (T1 + dir2 * lambda1[:, None]) + (T2 + dir2 * lambda2[:, None]) ) / 2
    
    @staticmethod
    def triangulate_pixels(
        pixels_1: np.ndarray,
        K_1: np.ndarray,
        dist_coeffs_1: np.ndarray,
        R_1: np.ndarray,
        T_1: np.ndarray,
        pixels_2: np.ndarray,
        K_2: np.ndarray,
        dist_coeffs_2: np.ndarray,
        R_2: np.ndarray,
        T_2: np.ndarray
        ) -> np.ndarray:
        """
        Parameters
        ----------
        pixels_1 : array_like
            list of N 2D pixel coordinates (shape Nx2) of camera 1.
        K_1 : array_like
            intrinsic matrix of camera 1.
        dist_coeffs_1 : array_like
            distortion coefficients of camera 1.
        R_1 : array_like
            rotation matrix (shape 3x3 or 3x1) of camera 1.
        T_1 : array_like
            translation vector (shape 3x1) of camera 1.
        pixels_2 : array_like
            list of N 2D pixel coordinates (shape Nx2) of camera 2.
        K_2 : array_like
            intrinsic matrix of camera 2.
        dist_coeffs_2 : array_like
            distortion coefficients of camera 2.
        R_2 : array_like
            rotation matrix (shape 3x3 or 3x1) of camera 2.
        T_2 : array_like
            translation vector (shape 3x1) of camera 2.
        
        Returns 
        ----------
        points3D_1
            numpy array (shape Nx3) of triangulated points viewed from camera 1
        points3D_2
            numpy array (shape Nx3) of triangulated points viewed from camera 2

        Notes
        -----
        pixels_1 and pixels_2 need to have the same shape. Additionally,
        pixels_1[i] and pixels_2[i] should be the corresponding 2D pixel
        coordinates of the same 3D object.
        To find the midpoint between the two lines for every point you
        want to triangulate, take the average between the tuple returned
        """
        
        rays1 = ThreeDUtils.camera_to_ray_world(pixels_1, R_1, T_1, K_1, dist_coeffs_1)
        rays2 = ThreeDUtils.camera_to_ray_world(pixels_2, R_2, T_2, K_2, dist_coeffs_2)

        origin1 = rays1[0].q
        origin2 = rays2[0].q
        directions1 = np.vstack([line.v for line in rays1])
        directions2 = np.vstack([line.v for line in rays2])

        lambda1 = ThreeDUtils.find_lambda(origin1, directions1, origin2, directions2)
        lambda2 = ThreeDUtils.find_lambda(origin2, directions2, origin1, directions1)

        # to find the midpoint between the two lines for every point, take the average between the two
        return origin1 + directions1 * lambda1, origin2 + directions2 * lambda2
    
    @staticmethod
    def intersect_pixels(
        pixels_1: np.ndarray,
        K_1: np.ndarray,
        dist_coeffs_1: np.ndarray,
        R_1: np.ndarray,
        T_1: np.ndarray,
        pixels_2: np.ndarray,
        shape_2: tuple,
        K_2: np.ndarray,
        dist_coeffs_2: np.ndarray,
        R_2: np.ndarray,
        T_2: np.ndarray,
        index: str='x'
        ) -> np.ndarray:
        """
        Parameters
        ----------
        pixels_1 : array_like
            list of N 2D pixel coordinates (shape Nx2) of camera 1.
        K_1 : array_like
            intrinsic matrix of camera 1.
        dist_coeffs_1 : array_like
            distortion coefficients of camera 1.
        R_1 : array_like
            rotation matrix (shape 3x3 or 3x1) of camera 1.
        T_1 : array_like
            translation vector (shape 3x1) of camera 1.
        pixels_2 : array_like
            list of N pixel coordinates (shape Nx1) of projector/camera 2.
        shape_2 : tuple
            resolution of projector/camera 2 given as (height, width)
        K_2 : array_like
            intrinsic matrix of projector/camera 2.
        dist_coeffs_2 : array_like
            distortion coefficients of projector/camera 2.
        R_2 : array_like
            rotation matrix (shape 3x3 or 3x1) of projector/camera 2.
        T_2 : array_like
            translation vector (shape 3x1) of projector/camera 2.
        index : str
            choice between 'x' and 'y' to say which coordinate from
            projector/camera 2 is avaialble, the other will be used
            to draw the line which will become a plane in the world 
        
        Returns 
        ----------
        points3D
            numpy array (shape Nx3) of triangulated points viewed from camera 1

        Notes
        -----
        pixels_1 and pixels_2 need to have the same length.
        pixels_2 has shape Nx1, while pixels_1 has shape Nx2, this is because
        we are taking a line across projector/camera 2 and projecting a plane
        onto the scene.
        If index=='x', we will take a vertical line across projector/camera 2,
        i.e. the line will have y going from 0 to height, while x stays constant.
        Otherwise, the line will have x going from 0 to width with y constant.
        """
        rays1 = ThreeDUtils.camera_to_ray_world(pixels_1, R_1, T_1, K_1, dist_coeffs_1)
        origin1 = rays1[0].q
        directions1 = np.vstack([line.v for line in rays1])
        height, width = shape_2

        lines_index_1 = np.stack([pixels_2,
                            np.zeros_like(pixels_2)], axis=1) if index=='x' else \
                        np.stack([np.zeros_like(pixels_2), pixels_2], axis=1)
        lines_index_2 = np.stack([pixels_2,
                            height * np.ones_like(pixels_2)], axis=1) if index=='x' else \
                        np.stack([width * np.ones_like(pixels_2), pixels_2], axis=1)
        rays_index_1 = ThreeDUtils.camera_to_ray_world(
            lines_index_1,
            R_2,
            T_2,
            K_2,
            dist_coeffs_2)
        rays_index_2 = ThreeDUtils.camera_to_ray_world(
            lines_index_2,
            R_2,
            T_2,
            K_2,
            dist_coeffs_2)
        directions_index_1 = np.vstack([line.v for line in rays_index_1])
        directions_index_2 = np.vstack([line.v for line in rays_index_2])
        normals = np.matmul(np.cross(directions_index_2, directions_index_1), R_2)
        origin = np.matmul(R_2.T, T_2)
                                
        L = np.matmul(normals, origin) / np.sum(np.multiply(directions1, normals), axis=1)

        return origin1 + directions1 * L[:, None]

    @staticmethod
    def triangulate_lines( lines: np.ndarray[Line] ) -> np.ndarray:
        """
        TODO: vectorize operations to avoid using list comprehension.
        This equation for the formula to find intersection of many lines 
        is described [here](https://en.wikipedia.org/wiki/Line–line_intersection#In_three_dimensions_2).

        Parameters
        ----------
        lines : array_like
            list of N > 1 lines, each containing a point and a direction vector.
            
        Returns 
        ----------
        closest point of intersection of all lines.
        """
        assert len(lines) > 1, "Need at least 2 lines to triangulate"
        As = [np.outer(line.v, line.v) - np.eye(3) for line in lines]
        Bs = [np.matmul(A, line.q.T).ravel() for A, line in zip(As, lines)]

        A = np.sum(np.stack(As, axis=2), axis=2)
        B = np.sum(np.stack(Bs, axis=1), axis=1)

        return (np.linalg.inv(A) @ B).reshape((1,3))

    @staticmethod
    def point_line_distance(p: np.ndarray, line: Line):
        """
        Finds distance between a point and a line.

        Parameters
        ----------
        p : array_like
            point of shape 1xN or Nx1. Function converts it to numpy array and to shape 1xN.
        line : Line
            line class containing a point and a direction vector.
            
        Returns 
        ----------
        distance between point and line
        """
        p = np.array(p, dtype=np.float32).reshape((1,-1))
        return np.linalg.norm(np.cross(line.v, p - line.q)) / np.linalg.norm(line.v)

    def camera_to_ray_world(
        points2D: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray
        ) -> np.ndarray[Line]:
        """
        Converts camera pixel coordinates in x,y to rays into the world.
        It additionally performs rotation and translation so that rays
        are in world coordinates.

        Parameters
        ----------
        points2D : array_like
            list of N 2D pixel coordinates
        R : array_like
            camera rotation in world coordinate system. If sent as a 3x1 or 1x3 vector,
            function uses Rodrigues to obtain 3x3 matrix
        T : array_like
            3x1 or 1x3 camera translation vector in world coordinate system
        K : array_like
            3x3 camera intrinsic matrix
        dist_coeffs : array_like 
            distortion coefficients of camera
        
        Returns
        ----------
        lines
            List of N lines in world coordinate system
        """
        R = np.array(R, dtype=np.float32)
        T = np.array(T, dtype=np.float32).reshape((3,1))
        if R.shape!=(3,3):
            R, _ = cv2.Rodrigues(R)
        uv = ImageUtils.undistort_camera_points(points2D, K, dist_coeffs)
        directions = ImageUtils.homogeneous_coordinates(uv)
        lines = [Line(ThreeDUtils.get_origin(R, T), np.matmul(R.T, direction.reshape((3,1)))) for direction in directions]
        return lines

    def camera_to_ray(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray=None
        ) -> np.ndarray[Line]:
        """
        Function provided for convenience.
        It differs from camera_to_ray_world() only in what argument(s) it accepts.

        Converts camera pixel coordinates in x,y to rays into the world.
        It does not perform rotation or translation, therefore it keeps the rays
        in camera coordinates where camera is origin and looking at [0,0,-1].

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
        return ThreeDUtils.camera_to_ray_world(points2D, np.zeros(3), np.zeros(3), K, dist_coeffs)

    def combine_transformations(
        R1: np.ndarray,
        T1: np.ndarray,
        R2: np.ndarray,
        T2: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        R1 : array_like
            First rotation matrix (shape 3x3)
        T1 : array_like
            First translation vector (shape 1x3 or 3x1).
        R2 : array_like
            Second rotation matrix (shape 3x3)
        T2 : array_like
            Second translation vector (shape 1x3 or 3x1).

        Returns
        ----------
        R
            Combined rotation matrix (shape 3x3)
        T
            Combined translation vector (shape 3x1)
        """
        R1 = np.array(R1, dtype=np.float32).reshape((3,3))
        T1 = np.array(T1, dtype=np.float32).reshape((3,1))
        R2 = np.array(R2, dtype=np.float32).reshape((3,3))
        T2 = np.array(T2, dtype=np.float32).reshape((3,1))

        R = np.matmul(R2, R1)
        T = np.matmul(R2, T1) + T2

        return R, T
    
    def get_origin(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Returns the origin of an object, which has tuple R and T, where
        R and T are the rotation and translation to bring world coordinates
        to the object coordinate system.

        Parameters
        ----------
        R : array_like
            rotation matrix (shape 3x3)
        T : array_like
            translation vector (shape 1x3 or 3x1).

        Returns
        ----------
        origin
            origin vector (shape 3x1) of object
        """
        R = np.array(R, dtype=np.float32).reshape((3,3))
        T = np.array(T, dtype=np.float32).reshape((3,1))
        return -np.matmul(R.T, T)
    
    @staticmethod
    def normals_from_point_cloud(points: np.ndarray) -> np.ndarray:
        """
        
        Parameters
        ----------
        points : array_like
            point cloud (shape Nx3)

        Returns
        -------
        normals
            numpy array (shape Nx3)
        """
        dx = (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / 1
        dy = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 1
        normals = - np.cross(dx, dy)
        # normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        return normals
    
    @staticmethod
    def normals_from_depth_map(depth_map: np.ndarray) -> np.ndarray:
        """       

        Parameters
        ----------
        depth_map : array_like
            depth map (shape Nx1)
            
        Returns
        -------
        normals
            numpy array (shape Nx3)
        """
        zy, zx = np.gradient(depth_map)
        normals = np.dstack((zx, zy, -np.ones_like(depth_map)))
        # n = np.linalg.norm(normals, axis=2)
        normals = normals / np.linalg.norm(normals, axis=2)[:, :, None]
        return normals
    
    @staticmethod
    def point_cloud_from_depth_map(
        depth_map: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        K: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        depth_map : array_like
            depth map (shape NxMx1) already undistorted
        R : array_like
            camera rotation in world coordinate system. If sent as a 3x1 or 1x3 vector,
            function uses Rodrigues to obtain 3x3 matrix
        T : array_like
            3x1 or 1x3 camera translation vector in world coordinate system
        K : array_like
            3x3 camera intrinsic matrix

        Returns
        -------
        point_cloud
            numpy array (shape Sx3) of points, where S = N * M

        Notes
        -----
        This function does not accept depth maps that have not been corrected for distortion.
        """
        R = np.array(R, dtype=np.float32)
        T = np.array(T, dtype=np.float32).reshape((1,3))
        if R.shape!=(3,3):
            R, _ = cv2.Rodrigues(R)

        pixel_x, pixel_y = pixel[0], pixel[1]

        depth = depth_map[pixel_y, pixel_x]
        
        result3D = np.matmul(np.linalg.inv(K), np.array([pixel_x, pixel_y, 1]))

        result3D *= depth

        result3D = np.matmul(R, result3D) + T

        return result3D

    @staticmethod
    def depth_map_from_point_cloud(points: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        point_cloud
            numpy array (shape Nx3) of points
            
        Returns
        -------
        depth_map : array_like
            depth map (shape Nx1)
        """
        pass

    @staticmethod
    def save_ply(
        filename: str,
        points: np.ndarray,
        normals: np.ndarray=None,
        colors: np.ndarray=None):
        """
        
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))

        #print(colors.shape, normals.shape, colors.dtype, normals.dtype)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float32))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        o3d.io.write_point_cloud(filename, pcd, compressed=False, print_progress=True)

    @staticmethod
    def load_ply(filename: str) -> np.ndarray:
        pc = o3d.io.read_point_cloud(filename, print_progress=True)
        return np.asarray(pc.points), np.asarray(pc.normals), np.asarray(pc.colors)