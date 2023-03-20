import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from colour_demosaicing import mosaicing_CFA_Bayer
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def mse2(img1, img2):
    """
    Calculate the mean squared error between two images
    :param img1:
    :param img2:
    :return:
    """
    mse_r = mean_squared_error(img1[:, :, 0], img2[:, :, 0])
    mse_g = mean_squared_error(img1[:, :, 1], img2[:, :, 1])
    mse_b = mean_squared_error(img1[:, :, 2], img2[:, :, 2])

    # Calculate the average MSE
    # return np.mean([mse_r, mse_g, mse_b])

    return np.array([mse_r, mse_g, mse_b])


# Define the size of the blocks
import itertools


# Read the image in grayscale format

# Dynamically crop image to a size that can be divided completely (ohne Rest)
def crop_image(img, block_size=8):
    """
    Crop the image to a size that can be divided completely
    :param img: image
    :param block_size:
    :return:
    """
    w, h, _ = img.shape
    w_crop = (w // 8) * block_size
    h_crop = (h // 8) * block_size

    img = img[0:w_crop, 0:h_crop, :]
    return img


# Divide the image into blocks
def divide_into_blocks(img, block_size):
    return [
        img[i : i + block_size, j : j + block_size]
        for i, j in itertools.product(
            range(0, img.shape[0], block_size),
            range(0, img.shape[1], block_size),
        )
    ]


# Calculate the variance of each block

def calc_variance(blocks):
    """
    Calculate the variance of a block
    :param block:
    :return:
    """
    return [np.var(block) for block in blocks]


# Define the threshold for identifying non-smooth blocks


# Mark non-smooth blocks
def mark_non_smooth_blocks(img, block_size: int = 8, threshold: float = 300) -> tuple:
    """
    :param img:
    :param block_size:
    :param threshold:
    :return:
    """
    non_smooth_blocks = []
    smooth_blocks = []
    for i, variance in enumerate(calc_variance(divide_into_blocks(img, block_size))):
        if variance < threshold:
            non_smooth_blocks.append(i)
        else:
            smooth_blocks.append(i)

    return non_smooth_blocks, smooth_blocks


# print(len(non_smooth_blocks))

# print(smooth_blocks)

# Create Error arrays for each of the 4 patterns with space for mse for each color channel


V_green = np.zeros(4)


def calc_uniformity(nsb, block_size=8):
    """
    Calculate the uniformity of vector V
    :param non_smooth_blocks: list of non-smooth blocks
    """
    U = np.zeros(len(nsb))
    E1 = np.zeros(3)
    E2 = np.zeros(3)
    E3 = np.zeros(3)
    E4 = np.zeros(3)

    E1_normed = np.zeros(3)
    E2_normed = np.zeros(3)
    E3_normed = np.zeros(3)
    E4_normed = np.zeros(3)
    for block in nsb:
        row = block // (img.shape[1] // block_size)
        col = block % (img.shape[1] // block_size)
        x1 = row * block_size
        y1 = col * block_size
        x2 = x1 + block_size
        y2 = y1 + block_size
        # Draw a rectangle around the non-smooth block in the original image
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # RGGB
        CFA_RGGB = mosaicing_CFA_Bayer(img[x1:x2, y1:y2], "RGGB")
        # BGGR
        CFA_BGGR = mosaicing_CFA_Bayer(img[x1:x2, y1:y2], 'BGGR')
        # GRBG
        CFA_GRBG = mosaicing_CFA_Bayer(img[x1:x2, y1:y2], 'GRBG')
        # GBRG
        CFA_GBRG = mosaicing_CFA_Bayer(img[x1:x2, y1:y2], 'GBRG')

        reipol_GBRG, reipol_GRBG, reipol_BGGR, reipol_RGGB = [demosaicing_CFA_Bayer_bilinear(CFA, pattern) for
                                                              CFA, pattern in
                                                              zip([CFA_GBRG, CFA_GRBG, CFA_BGGR, CFA_RGGB],
                                                                  ["GBRG", "GRBG", "BGGR", "RGGB"])]

        E1 = [E1[x] + mse2(img[x1:x2, y1:y2], reipol_RGGB)[x] for x in range(3)]
        E2 = [E2[x] + mse2(img[x1:x2, y1:y2], reipol_BGGR)[x] for x in range(3)]
        E3 = [E3[x] + mse2(img[x1:x2, y1:y2], reipol_GRBG)[x] for x in range(3)]
        E4 = [E4[x] + mse2(img[x1:x2, y1:y2], reipol_GBRG)[x] for x in range(3)]

        # Normalize mse against all mses of their pattern

        E1_normed = [E1_normed[x] + 100 * E1[x] / (E1[0] + E1[1] + E1[2]) for x in range(3)]

        E2_normed = [E2_normed[x] + 100 * E2[x] / (E1[0] + E1[1] + E1[2]) for x in range(3)]

        E3_normed = [E3_normed[x] + 100 * E3[x] / (E1[0] + E1[1] + E1[2]) for x in range(3)]

        E4_normed = [E4_normed[x] + 100 * E4[x] / (E1[0] + E1[1] + E1[2]) for x in range(3)]

        # Use only green channel - normalize green channel mse of a pattern against all other green channel mses

        V_green[0] += 100 * E1_normed[1] / (E1_normed[1] + E2_normed[1] + E3_normed[1] + E4_normed[1])
        V_green[1] += 100 * E2_normed[1] / (E1_normed[1] + E2_normed[1] + E3_normed[1] + E4_normed[1])
        V_green[2] += 100 * E3_normed[1] / (E1_normed[1] + E2_normed[1] + E3_normed[1] + E4_normed[1])
        V_green[3] += 100 * E4_normed[1] / (E1_normed[1] + E2_normed[1] + E3_normed[1] + E4_normed[1])

        U[block] = abs(V_green[0] - 25) + abs(V_green[1] - 25) + abs(V_green[2] - 25) + abs(V_green[3] - 25)
    return U

def evaluate_mcc(image, block_size=8, threshold=200):
    img = plt.imread(image)
    nsb, sb = mark_non_smooth_blocks(img, block_size, threshold)

    f = np.median(calc_uniformity(nsb))
    print(f)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Calculate uniformity of a given image')
    parser.add_argument('image', help='image to calculate uniformity of')
    parser.add_argument('-b', '--block-size', type=int, default=8, help='block size')
    parser.add_argument('-t', '--threshold', type=float, default=200, help='threshold for variance')
    args = parser.parse_args()

    img = plt.imread(args.image)
    nsb, sb = mark_non_smooth_blocks(img, args.block_size, args.threshold)

    f = np.median(calc_uniformity(nsb))
    print(f)