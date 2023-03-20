import cv2
import numpy as np
import pyjxl
from PIL import Image
import os
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

def confidence_to_binary(path_confidence_map, path_binary):
    # Load the confidence map
    confidence_map = cv2.imread(path_confidence_map, cv2.IMREAD_GRAYSCALE)

    # Apply threshold to the confidence map
    threshold_value = 0.5
    binary_map = np.zeros_like(confidence_map)
    binary_map[confidence_map > threshold_value * 255] = 255

    # Save the binary map to file
    cv2.imwrite(path_binary, binary_map)

def confidence_to_binary_adaptive(path_confidence_map, path_binary):
    # Load the confidence map
    confidence_map = cv2.imread(path_confidence_map, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive threshold to the confidence map
    binary_map = cv2.adaptiveThreshold(confidence_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save the binary map to file
    cv2.imwrite(path_binary, binary_map)


def compress_png_to_jpegxl(input_path, output_path):
    # Open the PNG image
    png_image = Image.open(input_path)

    # Convert to numpy array
    image_array = np.asarray(png_image)

    # Compress the image using PyJXL
    jxl_data = pyjxl.compress(image_array)

    # Save the compressed image to a file
    with open(output_path, "wb") as f:
        f.write(jxl_data)
    
def read_jpeg2000(filename):
    # Load the JPEG 2000 image using the OpenJPEG library
    with open(filename, "rb") as infile:
        data = infile.read()
        decoder = pyopenjpeg.ImageDecoder()
        image = decoder.decode(data)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the NumPy array to an OpenCV Mat object
    image_mat = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return image_mat

def open_j2k(input_path):
     # Open the JPEG 2000 image
    jpeg2000_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)#
    
    return jpeg2000_image

if __name__ == "__main__":
    #j2k_img = open_j2k("")
    image = cv2.imread("/Users/felixkurth/Documents/Inf_Master/PythonProgramme/MultimediaSecurity/MultimediaSecurityPS/CoMoFoD_sample/001_F_BC1_JP2K_Q95.jp2")
    cv2.imwrite('test.png', image)