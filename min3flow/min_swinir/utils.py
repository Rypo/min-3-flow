import os
import cv2
import numpy as np
import torch



def construct_weightname(task, scale=4, large_model=True, training_patch_size=None, noise=None, jpeg=None):
    '''
    001 Classical Image Super-Resolution (middle size)
     Note that --training_patch_size is just used to differentiate two different settings in Table 2 of the paper. Images are NOT tested patch by patch.
    
    (setting1: when model is trained on DIV2K and with training_patch_size=48)
     > 001_classicalSR_DIV2K_s48w8_SwinIR-M_x{2,3,4,8}.pth

    (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
     > 001_classicalSR_DF2K_s64w8_SwinIR-M_x{2,3,4,8}.pth
    
    002 Lightweight Image Super-Resolution (small size)
     > 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{2,3,4}.pth
    
    003 Real-World Image Super-Resolution (use --tile 400 if you run out-of-memory)
    (middle size)
     > 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x{2,4}_GAN.pth
    
    (larger size + trained on more datasets)
     > 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth

    004 Grayscale Image Deoising (middle size)
     > 004_grayDN_DFWB_s128w8_SwinIR-M_noise{15,25,50}.pth

    005 Color Image Deoising (middle size)
     > 005_colorDN_DFWB_s128w8_SwinIR-M_noise{15,25,50}.pth
    
    006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
     > 006_CAR_DFWB_s126w7_SwinIR-M_jpeg{10,20,30,40}.pth
    '''

    if task=='classical_sr':
        assert training_patch_size in [48, 64], 'training_patch_size must be in {48, 64}'
        assert scale in [2, 3, 4, 8], 'scale must be in {2, 3, 4, 8}'
        taskprefix = '001_classicalSR'
        data = 'DIV2K' if training_patch_size==48 else 'DF2K'
        train_patch_size=training_patch_size
        window = 8
        model_size = 'M'
        task_shortname = 'x'
        task_val = scale
        suffix = ''

    elif task=='lightweight_sr':
        assert scale in [2, 3, 4], 'scale must be in {2, 3, 4}'
        taskprefix = '002_lightweightSR'
        data = 'DIV2K'
        train_patch_size=64
        window = 8
        model_size = 'S'
        task_shortname = 'x'
        task_val = scale
        suffix = ''

    elif task=='real_sr':
        assert scale in [2, 4], 'scale must be in {2, 4}'
        taskprefix = '003_realSR_BSRGAN'
        data = 'DFOWMFC' if large_model else 'DFO'
        train_patch_size=64
        window = 8
        model_size = 'L' if large_model else 'M'
        task_shortname = 'x'
        task_val = scale if not large_model else 4
        suffix = '_GAN'

    elif task=='gray_dn':
        assert noise in [15, 25, 50], 'noise must be in {15, 25, 50}'
        taskprefix = '004_grayDN'
        data = 'DFWB'
        train_patch_size=128
        window = 8
        model_size = 'M'
        task_shortname = 'noise'
        task_val = noise
        suffix = ''

    elif task=='color_dn':
        assert noise in [15, 25, 50], 'noise must be in {15, 25, 50}'
        taskprefix = '005_colorDN'
        data = 'DFWB'
        train_patch_size=128
        window = 8
        model_size = 'M'
        task_shortname = 'noise'
        task_val = noise
        suffix = ''

    elif task=='jpeg_car':
        assert jpeg in [10, 20, 30, 40], 'jpeg must be in {10, 20, 30, 40}'
        taskprefix = '006_CAR'
        data = 'DFWB'
        train_patch_size=126
        window = 7
        model_size = 'M'
        task_shortname = 'jpeg'
        task_val = jpeg
        suffix = ''
    
    return f'{taskprefix}_{data}_s{train_patch_size}w{window}_SwinIR-{model_size}_{task_shortname}{task_val}{suffix}.pth'
    

def save_image(output, outpath):
    # save image
    if isinstance(output, torch.Tensor):
        output = output.detach_().squeeze().float().clamp_(0, 1).cpu().numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    if not outpath.endswith('.png'):
        outpath += '.png'

    cv2.imwrite(outpath, output)


def imread_norm(img_path, read_flag = cv2.IMREAD_COLOR):
    return cv2.imread(img_path, read_flag).astype(np.float32) / 255.

def get_image_pair(path, folder_lq, task, scale, noise, jpeg):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if task in ['classical_sr', 'lightweight_sr']:
        img_gt = imread_norm(path)
        img_lq = imread_norm(f'{folder_lq}/{imgname}x{scale}{imgext}')

    # 003 real-world image sr (load lq image only)
    elif task in ['real_sr']:
        img_gt = None
        img_lq = imread_norm(path)

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif task in ['gray_dn']:
        img_gt = imread_norm(path, cv2.IMREAD_GRAYSCALE)
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif task in ['color_dn']:
        img_gt = imread_norm(path)
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, noise / 255., img_gt.shape)

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif task in ['jpeg_car']:
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt.ndim != 2:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


# ---- Extract from: utils/util_calculate_psnr_ssim.py ----


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
