import cv2
import Imath
import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter
import OpenEXR

class ImageUtils:
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
    def load_ldr(filename: str, make_gray: bool=True, normalize: bool=True) -> np.ndarray:
        """
        Load LDR image using OpenCV. Flags can be set to make it grayscale
        and/or normalize the value range from [0, 2**16[ to [0, 1[.

        Parameters
        ----------
        filename : str
            path to image file.  
        make_gray : boolean, optional
            Flag to convert image to grayscale (True) or keep all three channels (False).
            Default is True.
        normalize : boolean, optional
            Flag to normalize image to grayscale (True) or keep all three channels (False).
            Default is True.
        """
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(filename, "does not exist!")

        if len(img.shape) == 3:
            img = img[:, :, ::-1]  # BGR by default, so convert it to RGB

        if make_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if normalize:
            img = img / 2**16

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
    def load_openexr(filename: str, make_gray: bool=True):
        """
        Load .EXR extension image using OpenEXR.

        Parameters
        ----------
        filename : str
            path to image file.  
        make_gray : boolean, optional
            Flag to convert image to grayscale (True) or keep all three channels (False).
            Default is True.
        """
        with open(filename, "rb") as f:
            in_file = OpenEXR.InputFile(f)
            try:
                dw = in_file.header()['dataWindow']
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                dim = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
                # print(dim)
                if len(in_file.header()['channels']) == 3:  # Scan
                    (r, g, b) = in_file.channels("RGB", pixel_type=pt)
                elif len(in_file.header()['channels']) >= 4:  # Sim
                    r = in_file.channel('color.R', pt)
                    g = in_file.channel('color.G', pt)
                    b = in_file.channel('color.B', pt)

                r = np.reshape(np.frombuffer(r, dtype=np.float32), dim)
                g = np.reshape(np.frombuffer(g, dtype=np.float32), dim)
                b = np.reshape(np.frombuffer(b, dtype=np.float32), dim)
                rgb = np.stack([r, g, b], axis=2)

                ret = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if make_gray else rgb

                return ret
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
