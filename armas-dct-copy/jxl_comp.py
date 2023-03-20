import os
from pathlib import Path
import subprocess

def compress_image_to_jpegxl(input_image_path, image_file, output_folder, compression_ratios):
    for ratio in compression_ratios:
        output_path = f"{output_folder}/{image_file}_{ratio}.jxl"
        subprocess.run(['cjxl', input_image_path, output_path, f"--quality={ratio}", "--effort=5"])


def is_non_binary_png(image_path):
    return image_path.endswith('.png') and '_B.png' not in image_path

if __name__ == "__main__":  
    input_folder = "/Users/felixkurth/Documents/Inf_Master/PythonProgramme/MultimediaSecurity/comofod_test"
    output_folder = "/Users/felixkurth/Documents/Inf_Master/PythonProgramme/MultimediaSecurity/jxl_comofod"
    compression_ratios = [80, 85, 90, 95, 100]

    input_path = Path(input_folder)

    for image_file in sorted(os.listdir(input_folder)):
        if is_non_binary_png(str(image_file)):
            input_image_path = f"/Users/felixkurth/Documents/Inf_Master/PythonProgramme/MultimediaSecurity/comofod_test/{image_file}"
            compress_image_to_jpegxl(input_image_path, image_file, output_folder, compression_ratios)
