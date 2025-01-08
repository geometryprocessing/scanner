import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from utils.image_utils import ImageUtils
from utils.three_d_utils import ThreeDUtils

def line(ax, p1, p2, *args, **kwargs):
    ax.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), np.array([p1[2], p2[2]]), *args, **kwargs)

def basis(ax, T, R, *args, length=1, **kwargs):
    T = T.flatten()
    Rx = np.array(R[:, 0]).flatten() # need array back to shape (3,)
    Ry = np.array(R[:, 1]).flatten() # need array back to shape (3,)
    Rz = np.array(R[:, 2]).flatten() # need array back to shape (3,)
    Tx = T + length * Rx
    Ty = T + length * Ry
    Tz = T + length * Rz

    line(ax, T, Tx, "b", **kwargs)
    ax.text(Tx[0], Tx[1], Tx[2], 'X', color='black')
    line(ax, T, Ty, "b", **kwargs)
    ax.text(Ty[0], Ty[1], Ty[2], 'Y', color='black')
    line(ax, T, Tz, "b", **kwargs)
    ax.text(Tz[0], Tz[1], Tz[2], 'Z', color='black')

def axis_equal_3d(ax, zoom=1):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r/zoom, ctr + r/zoom)
    plt.tight_layout()

class Plotter:
    @staticmethod
    def plot_distortion(image_shape: tuple,
                        K: np.ndarray,
                        dist_coeffs: np.ndarray,
                        figsize: tuple = (16,12),
                        filename: str = None):
        """
        Plot (with matplotlib) an image displaying lens distortion.
        
        Parameters
        ----------
        image_shape : tuple
            tuple containing (height, width) of image resolution.
        K : np.ndarray
            Intrinsic matrix of camera/projector.
        dist_coeffs : np.ndarray
            Distortion coefficients of camera/projector.
        figsize : tuple
            tuple containing (width, height) of figure to plot.
            The default is (16,12).
        filename : str
            if passed, path to file where figure will be saved.  
        """
        height, width = image_shape
        # Create a grid of points (pixel coordinates)
        x = np.linspace(0, width, 25)
        y = np.linspace(0, height, 25)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        # undistort points
        undistorted_points = ImageUtils.undistort_camera_points(points,
                                                     K, 
                                                     dist_coeffs,
                                                     P=K)
        # find displacement vectors from distortion
        vectors = undistorted_points - points

        plt.figure(figsize=figsize)
        plt.title('Lens Distortion')

        # center of image resolution
        plt.scatter(x=width/2, y=height/2,
                    s=min(width,height)/20, c='royalblue', marker='+')

        # central point found after calibration
        plt.scatter(x=K[0,2], y=K[1,2],
                    s=min(width,height)/20, c='royalblue', marker='o')
        
        # arrows displaying distortion magnitude in pixels
        plt.quiver(points[:,0], points[:,1], vectors[:,0], vectors[:,1],
                   angles='xy', scale_units='xy', scale=1, color='blue')

        # isolines of distortion magnitude in pixels
        contour = plt.tricontour(points[:,0], points[:,1], np.linalg.norm(vectors, axis=1),
                                 levels=[1.,2.,3.,4.,5.,6.,8.,10.,12.,16.,20.],
                                 colors='k')
        plt.clabel(contour, contour.levels, inline=True, fontsize=max(width,height)/100)

        # labels
        plt.xlabel('x (pixels)')
        plt.xlim([0, width])
        plt.ylabel('y (pixels)')
        plt.ylim([height, 0])
        plt.grid(visible=True)
        plt.show()
        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')

    @staticmethod
    def plot_markers(markers: np.ndarray,
                    image_shape: tuple,
                    K: np.ndarray,
                    figsize: tuple = (16,12),
                    filename: str = None):
        """
        Plot (with matplotlib) an image scattering the position
        of detected intrinsic markers used for calibration.

        Parameters
        ----------
        markers : array_like
            list of 2D markers in pixel coordinates.
        image_shape : tuple
            tuple containing (height, width) of image resolution.
        K : np.ndarray
            Intrinsic matrix of camera/projector.
        figsize : tuple
            tuple containing (width, height) of figure to plot.
            The default is (16,12).
        filename : str
            if passed, path to file where figure will be saved.  
        """
        height, width = image_shape

        plt.figure(figsize=figsize)
        plt.title('Intrinsic Markers')

        # center of image resolution
        plt.scatter(x=width/2, y=height/2,
                    s=min(width,height)/20, c='royalblue', marker='+')

        # central point found after calibration
        plt.scatter(x=K[0,2], y=K[1,2],
                    s=min(width,height)/20, c='royalblue', marker='o')
        
        # scatter intrinsic image points on the image
        points = np.concatenate(markers)
        plt.scatter(x=points[:,0], y=points[:,1],
                    s=min(width,height)/10, c='tab:green', alpha=0.5, edgecolors='k')
        
        # labels
        plt.xlabel('x (pixels)')
        plt.xlim([0, width])
        plt.ylabel('y (pixels)')
        plt.ylim([height, 0])
        plt.grid(visible=True)
        plt.show()
        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')
    
    @staticmethod
    def plot_errors():
        pass

    @staticmethod
    def plot_extrinsics(objects: list):
        """
        TODO: plot with open3d instead.
        
        TODO: Draw cameras in red, projectors in green.
              Show the focal length / FOV of objects 

        Plot (with matplotlib axes 3D) 3D position and orientation
        of a list of camera/projector objects.
        The objects need to be already calibrated, otherwise all
        of them will be plotted at the origin.
        """

        fig = plt.figure('3D Position')
        ax = fig.add_subplot(111, projection='3d')

        for idx, obj in enumerate(objects):
            # color = 'red' if isinstance(obj, Camera) else 'green'

            T = obj.get_translation()
            R = obj.get_rotation()
            origin = ThreeDUtils.get_origin(R, T).flatten()
            orientation = R.T
            # origin = T.flatten()
            basis(ax, origin, orientation, length=50)

            ax.text(origin[0], origin[1], origin[2], f'object_{idx}', color='red')

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        ax.view_init(0,0,0)

        axis_equal_3d(ax)

        return ax

        plt.show()
    def plot_decoding(camera_resolution: tuple,
                      index_x: np.ndarray=None,
                      index_y: np.ndarray=None,
                      cmap: str='jet',
                      figsize: tuple = (16,12),
                      filename: str = None):
        """
        Plot (with matplotlib) an image of decoded row (y) and column (x)
        indices from structured light pattern.
        It uses jet colormap by default.

        Parameters
        ----------
        camera_resolution : tuple
            tuple containing (height, width) of image resolution.
        projector_resolution : tuple
            tuple containing (height, width) of projector resolution.
        index_x : array_like
            numpy array of decoded column indices from structured light 
        cmap : str, optional
            string for indicating which colormap to use from matplotlib.
            The default is 'jet'.
        figsize : tuple, optional
            tuple containing (width, height) of figure to plot.
            The default is (16,12).
        filename : str, optional
            if passed, path to file where figure will be saved.  
        """
        cam_height, cam_width = camera_resolution

        # Plot index_x if provided
        if index_x is not None:
            plt.figure(figsize=figsize)
            plt.title("Decoded X Indices")
            plt.imshow(index_x, cmap=cmap, extent=[0, cam_width, cam_height, 0])
            plt.xlim([0, cam_width])
            plt.ylim([cam_height, 0])
            plt.axis('off')

            plt.show()

            if filename:
                plt.savefig(f"{filename}_column.png", transparent=True, bbox_inches='tight')

        # Plot index_y if provided
        if index_y is not None:
            plt.figure(figsize=figsize)
            plt.title("Decoded Y Indices")
            plt.imshow(index_y, cmap=cmap, extent=[0, cam_width, cam_height, 0])
            plt.xlim([0, cam_width])
            plt.ylim([cam_height, 0])  # Reverse y-axis
            plt.axis('off')

            plt.show()

            if filename:
                plt.savefig(f"{filename}_row.png", transparent=True, bbox_inches='tight')
    
    @staticmethod
    def plot_normal_map(normals: np.ndarray,
                        mask: np.ndarray,
                        figsize: tuple = (16,12),
                        filename: str = None):
        """
        Plot (with matplotlib) normals map as a 2D image.

        Parameters
        ----------
        normals : array_like
            array (shape Nx3) of normals
        mask : array_like
            array (shape HxW) of mask
        figsize : tuple, optional
            tuple containing (width, height) of figure to plot.
            The default is (16,12).
        filename : str, optional
            if passed, path to file where figure will be saved.  

        Notes
        -----
        Red encodes the X-component of the normal vector.
        Green encodes the Y-component of the normal vector.
        Blue encodes the Z-component of the normal vector.
        """
        image = np.repeat(np.zeros(shape=mask.shape)[:,:,np.newaxis], 3, axis=-1)
        image[mask] = normals

        image = .5 * (image + 1.)  # Convert from [-1, 1] to [0, 1]

        plt.figure(figsize=figsize)
        plt.title("Normals")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')

    @staticmethod
    def plot_depth_map(depth_map: np.ndarray,
                       cmap: str='turbo',
                       max_percentile: int=95,
                       min_percentile: int=5,
                       figsize: tuple = (12,16),
                       filename: str = None):
        """
        Plot (with matplotlib) depth map.
        It uses jet colormap by default.

        Parameters
        ----------
        depth_map : array_like
            array of depth map
        cmap : str, optional
            string for indicating which colormap to use from matplotlib.
            The default is 'turbo'. 
        max_percentile : int, optional
            top percentile at which to normalize the colormap
            default is 95
        min_percentile : int, optional
            bottom percentile at which to normalize the colormap
            default is 5
        figsize : tuple, optional
            tuple containing (width, height) of figure to plot.
            The default is (16,12).
        filename : str, optional
            if passed, path to file where figure will be saved.  
        """
        mask = abs(depth_map) > 1e-8
        disp_map = 1/depth_map
        disp_map[~mask] = 0
        print(disp_map)
        vmax = np.percentile(disp_map[mask], max_percentile)
        vmin = np.percentile(disp_map[mask], min_percentile)
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)
        mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
        image = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
        
        plt.figure(figsize=figsize)
        plt.title("Depth Map")
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')
