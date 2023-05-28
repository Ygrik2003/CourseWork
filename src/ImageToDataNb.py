import numpy as np
from tifffile import imread
from math import log
import numba as nb
import pandas as pd

from pymage_size import get_image_size

from Lib.texture import graycomatrix


methods = [
    "SHG Intensity Mean",
    "SHG Intensity MAD",
    "SHG Intensity Contrast",
    "SHG Intensity Correlation",
    "SHG Intensity Entropy",
    "SHG Intensity ASM",
    "SHG Intensity IDM",
    "R-Ratio Mean",
    "R-Ratio MAD",
    "R-Ratio Contrast",
    "R-Ratio Correlation",
    "R-Ratio Entropy",
    "R-Ratio ASM",
    "Degree of Circular Polarization Mean",
    "Degree of Circular Polarization MAD",
    "Degree of Circular Polarization Contrast",
    "Degree of Circular Polarization Correlation",
    "Degree of Circular Polarization Entropy",
    "Degree of Circular Polarization ASM",
    "Degree of Circular Polarization IDM",
    "SHG-CD MAD",
    "SHG-CD Contrast",
    "SHG-CD Correlation",
    "SHG-CD Entropy",
    "SHG-CD ASM",
    "SHG-CD IDM",
    "SHG-LD MAD",
    "SHG-LD Contrast",
    "SHG-LD Correlation",
    "SHG-LD Entropy",
    "SHG-LD ASM",
    "SHG-LD IDM",
    "Pixel Density"
]

path = "L:\\Projects\\DataForCourseWork\\montage1.tiff"

path = path
levels = 2 ** 16
width, height = get_image_size(path).get_dimensions()
fullImg = imread(path)
print(fullImg.shape)

img = ...
mu = ...
sigma = ...

def setImg(x, y, w, h):
    global img
    img = fullImg[x:x+w,y:y+h]


@nb.njit
def setP(offset_row, offset_col):
    offset_row = offset_row
    offset_col = offset_col
    P = graycomatrix(img, offset_row, offset_col, levels=levels, normed=True) 

    mu = np.array([0, 0])
    sigma = np.array([0, 0])
    
    mu[0] = np.sum(P[:, 0] * P[:, 2]) 
    mu[1] = np.sum(P[:, 1] * P[:, 2])
    
    
    sigma[0] = np.sqrt(np.sum(P[:, 2] * np.square(P[:, 0] - mu[0])))
    sigma[1] = np.sqrt(np.sum(P[:, 2] * np.square(P[:, 1] - mu[1])))

    return [
        img.mean() / img.max(), 
        np.median(np.abs(img - np.median(img))) / img.max(), 
        np.sum(P[:, 2] * np.square(P[:, 1] - P[:, 0])), 
        np.sum((P[:, 0] - mu[0]) * (P[:, 0] - mu[1]) * P[:, 2] / (sigma[0] * sigma[1])), 
        np.sum(-P * np.log(P)), np.sum(np.square(P[:, 2])), 
        np.sum(P[:, 2] / (1 + (P[:, 1] - P[:, 0]) ** 2))
    ]


try:
    for density in range(6, 7):
        data = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6'])
        for i in range(0, height - 2 ** density + 1, 2 ** density):
            for j in range(0, width - 2 ** density + 1, 2 ** density):
                setImg(i, j, 2 ** density, 2 ** density)
                data_row = np.array([])
                data_row = np.append(data_row, setP(1, 0))
                data_row = np.append(data_row, 4 ** density)
                data = pd.concat([data, pd.DataFrame(data_row.reshape((1, -1)), columns=['1', '2', '3', '4', '5', '6'])], ignore_index=True, axis=0)
                print(f'{round(100 * i / (height - 2 ** density), 2)}% {round(100 * j / (width - 2 ** density), 2)}%')
            
        data.to_excel(f"Data/mydata{density}.xlsx")
except KeyboardInterrupt as e:
    data.to_excel(f"Data/mydata000.xlsx")
