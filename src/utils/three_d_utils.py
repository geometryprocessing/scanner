import cv2
import numpy as np
import open3d as o3d

from utils.image_utils import ImageUtils

def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a list of N D-dimensional vectors.
    If a vector is all zero, it returns an array filled with np.nan in place of that vector.

    Parameters
    ----------
    vector : array_like
        vector of shape NxD

    Returns
    ----------
    array of normalized vector (along axis=1) of original shape NxD.
    """
    vector = np.array(vector)
    norm = np.linalg.norm(vector, axis=1).reshape((-1,1))
    return vector / norm

class ThreeDUtils:
    @staticmethod
    def fit_line( points: list ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit line through list of points using
        Singular Value Decomposition.

        Parameters
        ----------
        points : array_like
            list of N points (all points need to have the same number of dimensions).

        Returns
        ----------
        C
            point in line
        N
            vector (normalized in L-2 norm) direction of line 
        """
        points = np.array(points, dtype=np.float32)
        C = np.mean(points, axis=0)
        _, _, V = np.linalg.svd(points - C)
        N = V[0]
        return C, N / np.linalg.norm(N)
    
    @staticmethod
    def fit_plane( points: list ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit plane through list of points using
        Singular Value Decomposition.

        Parameters
        ----------
        points : array_like
            list of N points (all points need to have the same number of dimensions).

        Returns
        ----------
        C
            point in plane
        N
            normal vector (normalized in L-2 norm) of plane

        Notes
        -----
        the normal vector N and its opposite -N are both valid normals of a plane
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
        return C, N / np.linalg.norm(N)
    
    @staticmethod
    def intersect_line_with_plane(line_q: np.ndarray,
                                  line_v: np.ndarray,
                                  plane_q: np.ndarray, 
                                  plane_n: np.ndarray) -> np.ndarray:
        """
        Finds the point of intersection of a line with a plane.
        This function can also find:
        - the intersection of N lines with N planes, resulting in N points;
        - the intersection of N lines with 1 plane, resulting in N points.

        Parameters
        ----------
        line_q : array_like
            array (shape Nx3) of points from N lines
        line_v : array_like
            array (shape Nx3) of direction vectors from N lines
        plane_q : array_like
            array (shape Nx3) of points from N planes
        plane_n : array_like
            array (shape Nx3) of normal vectors from N planes
         
        Returns
        -------
        points
            array (shape Nx3) of points 

        Notes
        -----
        You can pass a single line_q for multiple line_v -- the function will
        assume that all the lines pass through that singular point.
        You can do the same with plane_q and plane_n.
        If you would like to find a single intersection point of N lines
        with a single plane, then you should take the average of the N
        points returned from this function to get back a singular point.
        """
        line_q = np.array(line_q).reshape((-1,3))
        line_v = normalize(np.array(line_v).reshape((-1,3)))
        plane_q = np.array(plane_q).reshape((-1,3))
        plane_n = normalize(np.array(plane_n).reshape((-1,3)))

        # if only one line point is passed but multiple line directions,
        # this function assumes that they all pass through that same point
        if line_q.shape[0] != line_v.shape[0]:
            print('line_q and line_v different shapes... tiling')
            line_q = np.tile(line_q, (line_v.shape[0],1))

        # if only one plane point is passed but multiple normal vectors,
        # this function assumes that the planes all pass through that same point
        if plane_q.shape[0] != plane_n.shape[0]:
            print('plane_q and plane_n different shapes... tiling')
            plane_q = np.tile(plane_q, (plane_n.shape[0],1))

        # if only one plane is passed, repeat it to have the same shape of
        # the input lines
        if plane_q.shape[0] != line_q.shape[0]:
            print('line_q and plane_q different shapes... tiling plane_q and plane_n')
            plane_q = np.tile(plane_q, (line_q.shape[0],1))
            plane_n = np.tile(plane_n, (line_q.shape[0],1))

        # Intersect a line(s) with a plane(s) (in 3D).
        L = np.einsum('ij,ij->i', (plane_q - line_q), plane_n) / np.einsum('ij,ij->i', plane_n, line_v)
        points = line_q + L.reshape((-1,1)) * line_v
        return points

    @staticmethod
    def intersect_line_with_line(q1: np.ndarray,
                                 v1: np.ndarray,
                                 q2: np.ndarray,
                                 v2: np.ndarray) -> np.ndarray:
        """
        Finds the point of intersection of a line with another line.
        This function can also find:
        - the intersection of N lines with N lines, resulting in N points;
        - the intersection of N lines with 1 line, resulting in N points.

        Parameters
        ----------
        q1 : array_like
            array (shape Nx3) of point from N lines
        v1 : array_like
            array (shape Nx3) of direction vector from N lines
        q2 : array_like
            array (shape Nx3) of point from N other lines
        v2 : array_like
            array (shape Nx3) of direction vector from N other lines
         
        Returns
        -------
        points
            array (shape Nx3) of points

        Notes
        -----
        You can pass a single q1 for multiple v1 -- the function will
        assume that all the lines pass through that singular point.
        You can do the same with q2 and v2.
        If you would like to find a single intersection point of N lines,
        do not use this function. Use instead intersect_lines(), which
        will return a single point for N lines.
        """
        q1 = np.array(q1).reshape((-1,3))
        v1 = normalize(np.array(v1).reshape((-1,3)))
        q2 = np.array(q2).reshape((-1,3))
        v2 = normalize(np.array(v2).reshape((-1,3)))

        # if only one line point is passed but multiple line directions,
        # this function assumes that they all pass through that same point
        if q1.shape[0] != v1.shape[0]:
            print('q1 and v1 different shapes... tiling')
            q1 = np.tile(q1, (v1.shape[0],1))

        # if only one plane point is passed but multiple normal vectors,
        # this function assumes that the planes all pass through that same point
        if q2.shape[0] != v2.shape[0]:
            print('q2 and v2 different shapes... tiling')
            q2 = np.tile(q2, (v2.shape[0],1))

        # if only one second line is passed, repeat it to have the same shape of
        # the input lines
        if q2.shape[0] != q1.shape[0]:
            print('q1 and q2 different shapes... tiling q2 and v2')
            q2 = np.tile(q2,(q1.shape[0], 1))
            v2 = np.tile(v2,(q1.shape[0], 1))

        L = ThreeDUtils.find_lambda(q1,v1,q2,v2)
        points = q1 + L.reshape((-1,1)) * v1
        return points

    @staticmethod
    def find_lambda(q1: np.ndarray,
                    v1: np.ndarray,
                    q2: np.ndarray,
                    v2: np.ndarray) -> np.ndarray:
        """
        Finds the length/depth (lambda) along the N lines

        Parameters
        ----------
        q1 : array_like
            array (shape Nx3) of point from N lines
        v1 : array_like
            array (shape Nx3) of direction vector from N lines
        q2 : array_like
            array (shape Nx3) of point from N other lines
        v2 : array_like
            array (shape Nx3) of direction vector from N other lines

        Returns 
        ----------
        L
            numpy array of lambdas (shape Nx1), where lambda is the depth along
            lines
        """
        v1v2 = np.einsum('ij,ij->i',v1, v2).reshape((-1,1))
        v1v1 = np.linalg.norm(v1, axis=1).reshape((-1,1))**2
        v2v2 = np.linalg.norm(v2, axis=1).reshape((-1,1))**2

        L = (  np.einsum('ij,ij->i',v1, q2 - q1).reshape((-1,1)) * v1v1 \
             + np.einsum('ij,ij->i',v2, q1 - q2).reshape((-1,1)) * v1v2) / (v1v1 * v2v2 - v1v2**2)

        # L = (np.matmul(directions1, origin2.T - origin1.T) * v1 + np.matmul(directions2, origin1.T - origin2.T) * v12) / (v1 * v2 - v12**2)
        return L

    @staticmethod
    def triangulate(line1_q: tuple | list | np.ndarray,
                    line1_n: tuple | list | np.ndarray,
                    line2_q: tuple | list | np.ndarray,
                    line2_n: tuple | list | np.ndarray, ) -> np.ndarray:
        """
        TODO: discard function

        Parameters
        ----------
        line1_q : array_like
            point in line 1
        line1_n : array_like
            direction vector of line 1 (will be normalized in L-2 norm)
        line2_q : array_like
            point in line 2
        line2_n : array_like
            direction vector of line 2 (will be normalized in L-2 norm)

        Returns 
        ----------
        triangulated point of two lines.
        """
        T1 = np.array(line1_q, dtype=np.float32).reshape((-1,1))
        T2 = np.array(line2_q, dtype=np.float32).reshape((-1,1))
        dir1 = normalize(np.array(line1_n, dtype=np.float32).reshape((-1,1)))
        dir2 = normalize(np.array(line2_n, dtype=np.float32).reshape((-1,1)))
        
        v12 = np.einsum('ij,ij->i',dir1, dir2)
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
        points3D
            numpy array (shape Nx3) of triangulated points viewed from camera 1

        Notes
        -----
        pixels_1 and pixels_2 need to have the same shape. Additionally,
        pixels_1[i] and pixels_2[i] should be the corresponding 2D pixel
        coordinates of the same 3D object.
        """
        
        origin1, rays1 = ThreeDUtils.camera_to_ray_world(pixels_1, R_1, T_1, K_1, dist_coeffs_1)
        origin2, rays2 = ThreeDUtils.camera_to_ray_world(pixels_2, R_2, T_2, K_2, dist_coeffs_2)
        return ThreeDUtils.intersect_line_with_line(origin1, rays1, origin2, rays2)
    
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
        origin1, rays1 = ThreeDUtils.camera_to_ray_world(pixels_1, R_1, T_1, K_1, dist_coeffs_1)

        height, width = shape_2
        lines_index_1 = np.stack([pixels_2,
                            np.zeros_like(pixels_2)], axis=-1) if index=='x' else \
                        np.stack([np.zeros_like(pixels_2), pixels_2], axis=-1)
        lines_index_2 = np.stack([pixels_2,
                            height * np.ones_like(pixels_2)], axis=-1) if index=='x' else \
                        np.stack([width * np.ones_like(pixels_2), pixels_2], axis=-1)
        lines_index_1 = lines_index_1.reshape((-1,2))
        lines_index_2 = lines_index_2.reshape((-1,2))

        _, rays_index_1 = ThreeDUtils.camera_to_ray_world(
            lines_index_1,
            R_2,
            T_2,
            K_2,
            dist_coeffs_2)
        _, rays_index_2 = ThreeDUtils.camera_to_ray_world(
            lines_index_2,
            R_2,
            T_2,
            K_2,
            dist_coeffs_2)
        
        normals = np.cross(rays_index_2, rays_index_1)

        return ThreeDUtils.intersect_line_with_plane(origin1, rays1, ThreeDUtils.get_origin(R_2, T_2), normals)
    @staticmethod
    def intersect_lines( lines: np.ndarray[tuple | list | np.ndarray] ) -> np.ndarray:
        """
        This equation for the formula to find the single intersection of many lines 
        is described [here](https://en.wikipedia.org/wiki/Line–line_intersection#In_three_dimensions_2).

        Parameters
        ----------
        lines : array_like
            list of N > 1 lines, each containing a point and a direction vector.
            This has shape Nx2, where at index i, lines[i][0] is the point and
            lines[i][1] is the direction vector.
            
        Returns 
        ----------
        closest point (shape 1x3) of intersection of all lines
        """
        assert len(lines) > 1, "Need at least 2 lines to triangulate"
        As = [np.outer(line[0], line[0]) - np.eye(3) for line in lines]
        Bs = [np.matmul(A, normalize(line[1]).T).ravel() for A, line in zip(As, lines)]

        A = np.sum(np.stack(As, axis=2), axis=2)
        B = np.sum(np.stack(Bs, axis=1), axis=1)

        return (np.linalg.inv(A) @ B).reshape((1,3))

    @staticmethod
    def point_line_distance(p: tuple | list | np.ndarray,
                            line_q: tuple | list | np.ndarray,
                            line_n: tuple | list | np.ndarray,) -> np.ndarray:
        """
        Finds distance between a point and a line.

        Parameters
        ----------
        p : array_like
            point of shape 1xN or Nx1. Function converts it to numpy array and to shape 1xN.
        line_q : array_like
            point in line
        line_n : array_like
            direction vector of line (will be normalized in L-2 norm)
            
        Returns 
        ----------
        distance (in L-2 norm) between point and line
        """
        p = np.array(p, dtype=np.float32).reshape((1,-1))
        return np.linalg.norm(np.cross(line_n, p - line_q)) / np.linalg.norm(line_n)

    def camera_to_ray_world(
        points2D: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray
        ) -> np.ndarray:
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
        origin
            center of projection (shape 1x3) in world coordinates 
        rays
            numpy array (shape Nx3) of rays in world coordinates
        """
        R = np.array(R, dtype=np.float32)
        T = np.array(T, dtype=np.float32).reshape((3,1))
        if R.shape!=(3,3):
            R, _ = cv2.Rodrigues(R)
        rays = ThreeDUtils.camera_to_ray(points2D, K, dist_coeffs)
        origin = ThreeDUtils.get_origin(R,T)

        # R.T @ ray, where ray is (3x1)
        return origin.reshape((1,3)), np.matmul(rays, R)

    def camera_to_ray(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray=None
        ) -> np.ndarray:
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
        rays
            numpy array of shape 
        """
        uv = ImageUtils.undistort_camera_points(points2D, K, dist_coeffs)
        rays = ImageUtils.homogeneous_coordinates(uv)
        return rays

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