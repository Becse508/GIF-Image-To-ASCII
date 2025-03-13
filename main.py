from os import listdir
from PIL import Image, ImageSequence
import numpy as np
from typing import Literal, Callable
from statistics import fmean


class ValueCalc:
    @staticmethod
    def __get_slice(img, x, y, shape):
        orig_shapey, orig_shapex = img.shape
        return img[round(y / shape[0] * orig_shapey) : round((y+1) / shape[0] * orig_shapey), round(x / shape[1] * orig_shapex) : round((x+1) / shape[1] * orig_shapex)].flatten()
    
    
    @staticmethod
    def none(img, x, y, shape) -> int:
        return img[y, x]
    
    @staticmethod
    def general(img: np.ndarray, x, y, shape) -> int:
        """The first pixel from the slice made by repositioning"""
        
        orig_shapey, orig_shapex = img.shape
        return img[round(y / shape[0] * orig_shapey), round(x / shape[1] * orig_shapex)]
    
    @staticmethod
    def average(img: np.ndarray, x, y, shape) -> int:
        """The average value of the slice"""
        
        return fmean(ValueCalc.__get_slice(img, x, y, shape))
        
    
    @staticmethod
    def average_nonzero(img: np.ndarray, x, y, shape) -> int:
        """The average value of the pixels in the slice. Ignores pixels with a value of 0. returns 0 if all pixels in the slice are zero"""
        
        sl = [i for i in ValueCalc.__get_slice(img, x, y, shape) if i != 0]
        return 0 if len(sl) == 0 else fmean(sl)
    
    
ValueFunc = Callable[[np.ndarray, int, int, tuple[int, int]], int]



charsets = {
    "shades": " ░▒▓█",
    "layers": " ▁▂▃▄▅▆▇█",
    "dots": " ⡀⡄⡆⡇⣇⣧⣷⣿",
    "symbolic1": " .,:;+*?%$#@",
    "symbolic2": " .:!*%$@&#SB",
    "brutal": """ .'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$""",
}


# SETTINGS
CHARSET = "shades" # one of 'charsets' OR custom
OUTPUT_MODE = "char" # only 'char' is available currently
OUTPUT_SIZE = (24, 32) # None for original size, ONLY DOWNSCALING WORKS
VALUECALC_FUNC = ValueCalc.average_nonzero # The function to use for calculating the value of a pixel when resizing the image
CROP_IMAGE = True # remove all empty rows and columns
OUT_SEPARATOR = "\n" # The separator to use between frames of GIF-s

INPATH = "input"
OUTPATH = "output"
# --------


CHARACTERS = charsets.get(CHARSET, CHARSET)


def load_frame(img: Image.Image, mode: str, crop: bool = True) -> np.ndarray:
    width, height = img.size
    # the 3rd dimension is for the color channels
    if mode == 'L':
        array = np.array(img.getdata(), int).reshape(height, width)
    else:
        array = np.array(img.getdata(), int).reshape(height, width, len(mode))
   
    if crop:
        array = array[~np.all(array == 0, axis=1)]
        array = np.delete(array, np.argwhere(np.all(array[..., :] == 0, axis=0)), axis=1)
        
    return array




def load_files(path: str, mode: Literal['L', 'RGB', 'RGBA'] = 'L') -> list[tuple[list[np.ndarray], str]]:
    
    filenames = listdir(path)
    print(f"loading {len(filenames)} file(s) from '{path}' .... ", end = "")
    
    out = []
    for fname in filenames:
        img = Image.open(f"{path}/{fname}")
        frame_list = []
        
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert(mode)
            frame_list.append(load_frame(frame, mode, CROP_IMAGE))
            
        out.append((frame_list, fname))
    
    print("DONE")
    return out


def convert(img: np.ndarray,
            mode: Literal['char', 'bgColor'],
            shape: tuple[int, int] = None,
            value_func: ValueFunc = ValueCalc.none) -> str:
    
    out = ""
    if shape is None: shape = img.shape
    
    if mode == 'char':
        for y in range(shape[0]):
            for x in range(shape[1]):
                out += CHARACTERS[round(value_func(img, x, y, shape) / 255 * len(CHARACTERS))]
            out += "\n"
    
    return out



for frame_list, name in load_files(INPATH):
    print(f"converting '{name}' to TEXT in '{OUTPUT_MODE}' mode .... ", end = "")
    
    with open(f"{OUTPATH}/{name}.txt", "w", encoding="utf-8") as f:
        f.write(OUT_SEPARATOR.join([convert(frame, OUTPUT_MODE, OUTPUT_SIZE, VALUECALC_FUNC) for frame in frame_list]))
        
    print("DONE")