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

class ImageToData:
    def __init__(self, path: str):
        self.methods = {
            "SHG Intensity" : [
                self.getMean,
                self.getMAD,
                self.getContrast,
                self.getCorrelation,
                self.getEntropy,
                self.getASM,
                self.getIDM
            ],
            "R-Ratio" : [
                self.getMean,
                self.getMAD,
                self.getContrast,
                self.getCorrelation,
                self.getEntropy,
                self.getASM
            ],
            "Degree of Circular Polarization" : [
                self.getMean,
                self.getMAD,
                self.getContrast,
                self.getCorrelation,
                self.getEntropy,
                self.getASM,
                self.getIDM
            ],
            "SHG-CD" : [
                self.getMAD,
                self.getContrast,
                self.getCorrelation,
                self.getEntropy,
                self.getASM,
                self.getIDM
            ],
            "SHG-LD" : [
                self.getMAD,
                self.getContrast,
                self.getCorrelation,
                self.getEntropy,
                self.getASM,
                self.getIDM
            ]
        }
        self.path = path
        self.levels = 2 ** 16
        self.width, self.height = get_image_size(path).get_dimensions()
        self.fullImg = imread(self.path)
        print(self.fullImg.shape)


    def setImg(self, x, y, w, h):
        self.img = self.fullImg[x:x+w,y:y+h]

    def setP(self, offset_row, offset_col):
        self.offset_row = offset_row
        self.offset_col = offset_col
        self.P = graycomatrix(self.img, offset_row, offset_col, levels=self.levels, normed=True) 

        self.mu = [
            np.sum(self.P[:, 0] * self.P[:, 2]), 
            np.sum(self.P[:, 1] * self.P[:, 2])
        ]
        self.sigma = [
            np.sqrt(np.sum(self.P[:, 2] * np.square(self.P[:, 0] - self.mu[0]))),
            np.sqrt(np.sum(self.P[:, 2] * np.square(self.P[:, 1] - self.mu[1])))
        ]

    def getMean(self):
        return self.img.mean() / self.img.max()

    def getMAD(self):
        return np.median(np.abs(self.img - np.median(self.img))) / self.img.max()

    def getContrast(self):
        return np.sum(self.P[:, 2] * np.square(self.P[:, 1] - self.P[:, 0]))
        
    def getCorrelation(self):
        return np.sum((self.P[:, 0] - self.mu[0]) * (self.P[:, 0] - self.mu[1]) * self.P[:, 2] / (self.sigma[0] * self.sigma[1]))
    
    def getEntropy(self):
        return np.sum(-self.P * np.log(self.P))

    def getASM(self):
        return np.sum(np.square(self.P[:, 2]))

    def getIDM(self):
        return np.sum(self.P[:, 2] / (1 + (self.P[:, 1] - self.P[:, 0]) ** 2))

    def getMethodResult(self, method):
        for param_func in self.methods[method]:
            yield param_func()

itd = ImageToData("L:\\Projects\\DataForCourseWork\\montage1.tiff")
# itd = ImageToData("/home/ygrik/shared/Projects/DataForCourseWork/montage1.tiff")
# itd = ImageToData("D:/DataForCourseWork/montage1.tiff")

data = pd.DataFrame(columns=methods)

try:
    for density in range(6, 7):
        for i in range(0, itd.height - 2 ** density + 1, 2 ** density):
            print(f'{round(100 * i / (itd.height - 2 ** density), 2)}%')
            for j in range(0, itd.width - 2 ** density + 1, 2 ** density):
                itd.setImg(i, j, 2 ** density, 2 ** density)
                itd.setP(1, 0)
                data_row = np.array([])
                for key in itd.methods.keys():
                    data_row = np.append(data_row, list(itd.getMethodResult(key)))
                data_row = np.append(data_row, 4 ** density)

                data = pd.concat([data, pd.DataFrame(data_row.reshape((1, -1)), columns=methods)], ignore_index=True, axis=0)
            
        data.to_excel(f"Data/mydata{density}.xlsx")
except KeyboardInterrupt as e:
    data.to_excel(f"Data/mydata000.xlsx")
