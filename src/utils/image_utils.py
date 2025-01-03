import cv2
import Imath
import numpy as np
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
        TODO: understand better the functionality of cv2.undistortPoints
        In order to find where the points would be from undistortion, pass the same matrix K for P.
        """
        return cv2.undistortPoints(np.array(points2D, dtype=np.float32).reshape((-1, 1, 2)), K, dist_coeffs, R, P).reshape((-1, 2))
    
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
            List of N 2D homogenous coordinates (shape Nx3).
        """
        points = np.array(points2D, dtype=np.float32)
        return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

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
        img = image.deepcopy()
        img = ((img / np.percentile(img, q=99.99)) * (2**16 - 1)).astype(np.uint16)

        return img
    
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
        img = image.deepcopy()
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
        img = image.deepcopy()
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
        if type(source) is str:
            with open(source, 'rb') as f:
                raw = rawpy.imread(f)
        else:
            raw = rawpy.imread(source)

        bayer = raw.raw_image_visible.astype(np.float32)
        bayer -= 512
        bayer /= 16372-512
        return np.maximum(0, np.minimum(bayer, 1))
    
    @staticmethod
    def load_ldr(filename: str, make_gray: bool=False, normalize: bool=False) -> np.ndarray:
        """
        Load LDR (low dynamic range) image using OpenCV. Flags can be set to make it grayscale
        and/or normalize the value range from [0, 2**16[ to [0, 1[.

        Parameters
        ----------
        filename : str
            path to image file.  
        make_gray : boolean, optional
            Flag to convert image to grayscale (True) or keep all three color channels intact (False).
            Default is False.
        normalize : boolean, optional
            Flag to normalize image range to single [0, 1[ (True) or keep as uint16 [0, 2**16[ (False).
            Default is False.
        """
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(filename, "does not exist!")

        if len(img.shape) == 3:
            img = img[:, :, ::-1]  # BGR by default, so convert it to RGB

        if make_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # if normalize:
            ## why do we divide by 2**16?
            # img = img / 2**16

        # if chronos_raw:
        #     if len(img.shape) == 2:
        #         # if np.max(img) >= 2**15:
        #         #     print(np.max(img))
        #         img = img / 2**16
        #         img = self.demosaic(img, roll=True)
        #         if wb is not None:
        #             img[:, :, 0] *= wb[0]
        #             img[:, :, 2] *= wb[2]
        #     # print(img.shape, img.dtype)
        #     for i in range(3):
        #         pass
        #         # print(i, np.min(img[:, :, i]), np.max(img[:, :, i]))
        #         # plt.figure(str(i))
        #         # plt.hist(img[:, :, i].ravel(), bins=2**12)

        return img
    
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
        with open(filename, "rb") as f:
            in_file = OpenEXR.InputFile(f)
            try:
                dw = in_file.header()['dataWindow']
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
                if len(in_file.header()['channels']) == 3:
                    (r, g, b) = in_file.channels("RGB", pixel_type=pt)
                    d = None
                elif len(in_file.header()['channels']) >= 4:
                    r = in_file.channel('color.R', pt)
                    g = in_file.channel('color.G', pt)
                    b = in_file.channel('color.B', pt)

                    if load_depth:
                        d = in_file.channel("distance.Y", pt)
                        d = np.reshape(np.frombuffer(d, dtype=np.float32), dim)
                    else:
                        d = None

                r = np.reshape(np.frombuffer(r, dtype=np.float32), dim)
                g = np.reshape(np.frombuffer(g, dtype=np.float32), dim)
                b = np.reshape(np.frombuffer(b, dtype=np.float32), dim)
                rgb = np.stack([r, g, b], axis=2)

                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if make_gray else rgb
                if load_depth:
                    return img, d
                else:
                    return img
            finally:
                in_file.close()

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
                R = G = B = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float16).tobytes()
            else:
                R = image[:, :, 0].astype(np.float16).tobytes()
                G = image[:, :, 1].astype(np.float16).tobytes()
                B = image[:, :, 2].astype(np.float16).tobytes()
        else:
            R = G = B = image.astype(np.float16).tobytes()

        header = OpenEXR.Header(image.shape[1], image.shape[0])
        header['Compression'] = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)
        header['channels'] = {'R': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                            'G': Imath.Channel(Imath.PixelType(OpenEXR.HALF)),
                            'B': Imath.Channel(Imath.PixelType(OpenEXR.HALF))}

        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'R': R, 'G': G, 'B': B})   # need to duplicate channels for grayscale anyways
                                                    # (to keep it readable by LuminanceHDR)
        exr.close()

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
    