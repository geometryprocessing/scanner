import cv2
from copy import deepcopy
import Imath
import numpy as np
from PIL import Image, ExifTags
import rawpy
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_erosion
import OpenEXR

class ImageUtils:
    @staticmethod
    def undistort_camera_points(
        points2D: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        R: np.ndarray=None,
        P: np.ndarray=None ) -> np.ndarray:
        """
        This function, by default, takes a array of 2D pixel coordinates (x,y)
        and returns normalized, homogenous coordinates (u,v,1).

        Parameters
        ----------
        points2D : array_like
            array (Nx2) of 2D pixel coordinates
        K : array_like
            3x3 camera intrinsic matrix
        dist_coeffs : array_like
            distortion coefficients of camera
        R : array_like, optional
            rotation vector (3x1)
        P : array_like, optional
            new camera matrix (3x3) or new projection matrix (4x3).
            If you pass P=K, this function returns the undistorted points back in 
            (x,y) pixel coordinates
        """
        return cv2.undistortPoints(np.array(points2D, dtype=np.float32).reshape((-1, 1, 2)), K, dist_coeffs, R, P).reshape((-1, 2))
    
    @staticmethod
    def undistort_image(
        image: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        newK : np.ndarray=None
        ) -> np.ndarray:
        """
        Take image as numpy array and undistort it using OpenCV.
        Function creates a copy of image and returns the undistorted version.

        Parameters
        ----------
        image : array_like
            image array
        K : array_like
            3x3 camera intrinsic matrix
        dist_coeffs : array_like
            distortion coefficients of camera
        newK : array_like, optional
            3x3 new camera intrinsic matrix

        Returns
        ----------
        undistorted copy of image
        """
        return cv2.undistort(deepcopy(image), K, dist_coeffs, newK)


    @staticmethod
    def homogeneous_coordinates(
        points2D: np.ndarray
        ) -> np.ndarray:
        """
        Converts coordinates into homogenous form, i.e.
        the point (u, v) becomes (u, v, 1)

        Parameters
        ----------
        points2D : array_like
            list of N 2D coordinates (shape Nx2).
        
        Returns
        ----------
        homegeneous coordinates
            array (shape Nx3) of homogenous coordinates
        """
        points = np.array(points2D, dtype=np.float32).reshape((-1,2))
        return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).reshape((-1,3))

    @staticmethod
    def colorize(image: np.ndarray) -> np.ndarray:
        """
        Take LDR image and colorize it.
        Function creates a copy of image and returns the colorized version.

        Parameters
        ----------
        image : array_like
            image array

        Returns
        ----------
        colorized copy of image
        """
        img = deepcopy(image)
        img = ((img / np.percentile(img, q=99.99)) * (2**16 - 1)).astype(np.uint16)

        return img
    
    @staticmethod
    def normalize_color(
        color_image: str | np.ndarray,
        white_image: str | np.ndarray,
        black_image: str | np.ndarray = None
        ) -> np.ndarray:
        """
        Take a color image and a white image, apply a Gaussian blur
        on both and get color divided by white.
        This function is relevant to LookUp reconstruction.

        Parameters
        ----------
        color_image : array_like or string
            image when color pattern is projected onto the scene
        white_image : array_like or string
            image when white flash is projected onto the scene

        Returns
        -------
        normalized
            numpy array of color_image / white_image
        """
        if isinstance(white_image, str):
            white_image = ImageUtils.load_ldr(white_image)
        if isinstance(color_image, str):
            color_image = ImageUtils.load_ldr(color_image)
        if black_image is not None and isinstance(black_image, str):
            black_image = ImageUtils.load_ldr(color_image)

        white_image = ImageUtils.blur_3(white_image, sigmas=np.ones(3) * 1.5)
        color_image = ImageUtils.blur_3(color_image, sigmas=np.ones(3) * 1.5)

        thr, s = 0.01, np.min(white_image, axis=2)
        mask = s > thr * np.max(s)

        normalized = np.zeros_like(color_image, dtype=np.float32)

        if black_image is not None:
            black_image = ImageUtils.blur_3(black_image, sigmas=np.ones(3) * 1.5)
            normalized[mask] = (color_image[mask] - black_image[mask]) / (white_image[mask] - black_image[mask])
        else:
            normalized[mask] = color_image[mask] / white_image[mask]

        return normalized

    
    @staticmethod
    def demosaic(bayer: np.ndarray, roll: bool=False) -> np.ndarray:
        """
        Take raw image and demosaic it, i.e. reconstruct full color image.
        Function creates a copy of image and returns the demosaiced version.

        Parameters
        ----------
        bayer : array_like
            image array of Bayer
        roll : boolean, optional
            Flag to 'roll' (shift array elements, check numpy.roll() for more information)
            the pixels in the column-space of array by 1 position.
            Default is False.

        Returns
        ----------
        demosaiced image
        """
        rgb = np.zeros((bayer.shape[0], bayer.shape[1], 3), dtype=np.float32)
        if roll:
            bayer = np.roll(bayer, 1, axis=1)

        r = bayer[::2, ::2]
        rgb[::2, ::2, 0] = r
        rgb[::2, 1:-2:2, 0] = 0.5*(r[:, :-1] + r[:, 1:])
        rgb[1:-2:2, ::2, 0] = 0.5*(r[:-1, :] + r[1:, :])
        rgb[1:-2:2, 1:-2:2, 0] = 0.25*(r[:-1, :-1] + r[:-1, 1:] + r[1:, :-1] + r[1:, 1:])

        rgb[:,:,1] = bayer
        rgb[1:-2:2, 1:-2:2, 1] = 0.25*(bayer[1:-2:2, :-2:2] + bayer[1:-2:2, 2::2] + bayer[:-2:2, 1:-2:2] + bayer[2::2, 1:-2:2])
        rgb[2::2, 2::2, 1] = 0.25*(bayer[2::2, 1:-2:2] + bayer[2::2, 3::2] + bayer[1:-2:2, 2::2] + bayer[3::2, 2::2])

        b = bayer[1::2, 1::2]
        rgb[1::2, 1::2, 2] = b
        rgb[1::2, 2::2, 2] = 0.5*(b[:, :-1] + b[:, 1:])
        rgb[2::2, 1::2, 2] = 0.5*(b[:-1, :] + b[1:, :])
        rgb[2::2, 2::2, 2] = 0.25*(b[:-1, :-1] + b[:-1, 1:] + b[1:, :-1] + b[1:, 1:])

        return rgb
    
    @staticmethod
    def blur_3(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
        """
        Take image and apply Gaussian blur kernel.
        Function creates a copy of image and returns the blurred version.

        Parameters
        ----------
        image : array_like
            image array
        sigmas : list[float]
            list of sigma (float) for Gaussian filter.
            Sigma values will be applied to each image channel in the order they are passed.

        Returns
        ----------
        blurred copy of image
        """
        img = deepcopy(image)
        for i in range(3):
            img[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigmas[i])

        return img
    
    @staticmethod
    def replace_hot_pixels(image, dark, thr=32):
        """
        Take image and replace hot pixels with neighboring value.

        Parameters
        ----------
        image : array_like
            image array
        dark : array_like
            image array of the scene when no active lighting is applied
        thr : int
            threshold value to consider a pixel 'hot'.

        Returns
        ----------
        copy of image with hot pixels replaced
        """
        img = deepcopy(image)
        h, w = img.shape[:2]
        rr, cc = np.nonzero(dark > thr)

        for r, c in zip(rr, cc):
            v, n = 0, 0
            if c > 0:
                v += img[r, c - 1]
                n += 1
            if c < w - 1:
                v += img[r, c + 1]
                n += 1
            if r > 0:
                v += img[r - 1, c]
                n += 1
            if r < h - 1:
                v += img[r + 1, c]
                n += 1
            img[r, c] = v / n

        # print(f"Replaced {rr.shape[0]} hot/stuck pixels with average value of their neighbours")

        return img
    
    @staticmethod
    def load_arw(source):
        if isinstance(source, str):
            with open(source, 'rb') as f:
                raw = rawpy.imread(f)
        else:
            raw = rawpy.imread(source)

        bayer = raw.raw_image_visible.astype(np.float32)
        bayer -= 512
        bayer /= 16372-512
        return np.maximum(0, np.minimum(bayer, 1))
    
    @staticmethod
    def load_ldr(filename: str, make_gray: bool = False, normalize: bool = False) -> np.ndarray:
        """
        Load LDR (low dynamic range) image using Pillow. Flags can be set to make it grayscale
        and/or normalize the value range to [0, 1.0) as np.float64.

        Parameters
        ----------
        filename : str
            Path to the image file.  
        make_gray : bool, optional
            Flag to convert the image to grayscale (True) or keep all color channels intact (False).
            Default is False.
        normalize : bool, optional
            Flag to normalize image range to single [0, 1[ (True) or keep as uint16 [0, 2**16[ (False).
            Default is False.

        Returns
        -------
        np.ndarray
            Loaded image as a NumPy array, with optional grayscale conversion and normalization applied.

        Notes
        -----
        This function ignores the EXIF orientation tags.
        """
        try:
            # Open the image with Pillow
            img = Image.open(filename)

            # # Handle EXIF orientation
            # try:
            #     for orientation in ExifTags.TAGS.keys():
            #         if ExifTags.TAGS[orientation] == "Orientation":
            #             break

            #     exif = img._getexif()
            #     if exif is not None and orientation in exif:
            #         if exif[orientation] == 3:
            #             img = img.rotate(180, expand=True)
            #         elif exif[orientation] == 6:
            #             img = img.rotate(270, expand=True)
            #         elif exif[orientation] == 8:
            #             img = img.rotate(90, expand=True)
            # except Exception as e:
            #     print(f"Warning: Could not handle EXIF orientation: {e}")

            img_array = np.array(img)
            # save the data type and shape for future operations
            dtype = img_array.dtype
            shape = img_array.shape

            # Convert image to grayscale if requested using the first three channels
            if make_gray and len(shape) > 2 and shape[2] >= 3:
                img_array = (.2989 * img_array[:,:,0] \
                             + .5870 * img_array[:,:,1] \
                                + .1140 * img_array[:,:,2]).astype(dtype)

            # Normalize pixel values if requested
            if normalize:
                m = np.iinfo(dtype).max if dtype.kind in 'iu' else np.finfo(dtype).max
                img_array = (img_array / m).astype(np.float64)

            return img_array

        except FileNotFoundError:
            print(f"{filename} does not exist.")
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    
    @staticmethod
    def save_ldr(filename: str, image: np.ndarray, ensure_rgb: bool=False):
        """
        Save image with .LDR extension image using OpenCV.

        Parameters
        ----------
        filename : str
            path to file where image will be saved.  
        image : array_like
            image array (if only one channel, i.e. grayscale, keeps grayscale)
        keep_rgb : boolean, optional
            Flag to convert RGB image to grayscale (False) or save with all three channels (True).
            Default is False.
        """
        if len(image.shape) == 2 and ensure_rgb:
            image = np.repeat(image[:, :, None], 3, axis=2)

        if len(image.shape) > 2:
            image = image[:, :, ::-1]  # RGB to BGR for cv2 (if color)

        cv2.imwrite(filename, image)  # expects BGR or Gray
    
    @staticmethod
    def load_openexr(filename: str, make_gray: bool=False, load_depth: bool=False):
        """
        Load .EXR extension image using OpenEXR. These are multi-channel,
        high-dynamic range images.

        Parameters
        ----------
        filename : str
            path to image file.  
        make_gray : boolean, optional
            Flag to convert image to grayscale (True) or keep all three color channels intact (False).
            Default is False.
        load_depth : boolean, optional
            Flag to load the channel distance.Y (True) or not (False).
            Default is False.

        Returns
        -------
        image
            numpy array (shape NxMx3 if make_gray set to False,
            shape NxMx1 if make_gray set to True) of image
        depth (optional)
            numpy array (shape NxMx1) of depth (only returned if load_depth set to True)
        """
        with OpenEXR.File(filename) as infile:
            dw = infile.header()['dataWindow']
            dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            if len(infile.header()['channels']) == 3:
                (R, G, B) = infile.channels()["RGB"]
                d = None
            elif len(infile.header()['channels']) >= 4:
                R = infile.channels()["R"].pixels
                G = infile.channels()["G"].pixels
                B = infile.channels()["B"].pixels

                if load_depth:
                    d = infile.channel()["distance.Y"].pixels
                else:
                    d = None

            rgb = np.stack([R, G, B], axis=2)

            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if make_gray else rgb
            if load_depth:
                return img, d
            else:
                return img

    @staticmethod
    def save_openexr(filename: str, image: np.ndarray, make_gray: bool=True):
        """
        Save image with .EXR extension using OpenEXR.

        Parameters
        ----------
        filename : str
            path to file where image will be saved.  
        image : array_like
            image array (if only one channel, i.e. grayscale, keeps grayscale)
        make_gray : boolean, optional
            Flag to convert RGB image to grayscale (True) or save with all three color channels (False).
            Default is True.
        """
        if len(image.shape) > 2:
            if make_gray:
                R = G = B = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float16)
            else:
                R = image[:, :, 0].astype(np.float16)
                G = image[:, :, 1].astype(np.float16)
                B = image[:, :, 2].astype(np.float16)
        else:
            R = G = B = image.astype(np.float16)

        header = { 'compression' : OpenEXR.PXR24_COMPRESSION}
                #   "type" : OpenEXR.scanlineimage }
        channels = {'R': R,
                    'G': G,
                    'B': B}

        with OpenEXR.File(header, channels) as outfile:
            outfile.write(filename)

    @staticmethod
    def linear_map(img, thr=None, mask=None, gamma=1.0):
        """
        """
        if thr is None:
            pixels = img[mask].ravel() if mask is not None else img.ravel()

            if pixels.shape[0] > 1e+6:
                pixels = pixels[::int(pixels.shape[0] / 1e+6)]

            thr = 1.2 * np.sort(pixels)[int(0.99*pixels.shape[0])]  # threshold at 99th percentile

        img = img / thr
        if abs(gamma - 1.0) > 1e-6:
            img = np.power(img, 1/gamma)

        return np.minimum(255 * img, 255).astype(np.uint8), thr

    @staticmethod
    def generate_mask(image: np.ndarray,
                      threshold: float,
                      mask_sigma: float=3,
                      rank: int=2,
                      connectivity: int=1,
                      iterations: int=6):
        """
        Take in an image and generate a binary mask from it using scipy.ndimage

        Parameters
        ----------
        image : array_like
            image for which a binary mask will be generated
        threshold : float
            ls
        mask_sigma : scalar or sequence of scalars, optional
            Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a
            sequence, or as a single number, in which case it is equal for
            all axes.
            Default is set to 3 for all axes.
        rank : int, optional
            Number of dimensions of the array to which the structuring element
            will be applied, as returned by `np.ndim`.
            Default is set to 2 (a square).
        connectivity : int, optional
            `connectivity` determines which elements of the output array belong
            to the structure, i.e., are considered as neighbors of the central
            element. Elements up to a squared distance of `connectivity` from
            the center are considered neighbors. `connectivity` may range from 1
            (no diagonal elements are neighbors) to `rank` (all elements are
            neighbors).
            Default is set to 1 (only immediate neighbors).
        iterations : int, optional
            The erosion is repeated `iterations` times (one, by default).
            If iterations is less than 1, the erosion is repeated until the
            result does not change anymore.
            Default is set to 6.
        
        Returns
        -------
        mask
            numpy array of binary mask
        
        """
        # blur image
        ldr, thr_ldr = ImageUtils.linear_map(gaussian_filter(image, sigma=mask_sigma))
        # use OTSU for thresholding (avoids setting a fixed value)
        thr_otsu, mask = cv2.threshold(ldr, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print("Thresholds:", thr_ldr, thr_otsu)
        # generate binary structure and erode mask
        struct = generate_binary_structure(rank, connectivity)
        mask = binary_erosion(mask, struct, iterations)

        return mask
    