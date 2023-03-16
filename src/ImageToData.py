import numpy as np
from tifffile import imread
from math import log
import numba as nb

from pymage_size import get_image_size

from Lib.texture import graycomatrix

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
        

    def getMean(self):
        return self.img.mean() / self.img.max()

    def getMAD(self):
        return np.median(np.abs(self.img - np.median(self.img))) / self.img.max()

    def getContrast(self):
        def contrast(P): # P = [i, j, p]
            return P[2] * (P[1] - P[0]) ** 2
        
        contrastVectorize = np.vectorize(contrast, signature='(3)->()')
        return np.sum(contrastVectorize(self.P))
        

    def getCorrelation(self):
        # def getMu(p, axis):
        #     def getLocalMu(x):
        #         if axis == 0:
        #             return np.sum(x * self.P[x, :])
        #         elif axis == 1:
        #             return np.sum(x * self.P[:, x])
        #     getLocalMuVectorize = np.vectorize(getLocalMu, signature="(3)->()")
        #     return np.sum(getLocalMuVectorize(p))
        

        def getSigma(axis):
            pass

    def getEntropy(self):
        def entropy(p):
            return -p * np.log(p, dtype=np.float16)
        return np.sum(entropy(self.P[:, 2]))

    def getASM(self):
        return np.sum(np.square(self.P[:, 2]))

    def getIDM(self):
        def IDM(P): # P = [i, j, p]
            return P[2] / (1 + (P[1] - P[0]) ** 2)
        
        IDMVectorize = np.vectorize(IDM, signature='(3)->()')
        
        idm = np.sum(IDMVectorize(self.P))
        return idm

    def getMethodResult(self, method):
        for param_func in self.methods[method]:
            yield param_func()

# itd = ImageToData("L:\\Projects\\DataForCourseWork\\montage1.tiff")
# itd = ImageToData("/home/ygrik/shared/Projects/DataForCourseWork/montage1.tiff")
itd = ImageToData("../DataForCourseWork/montage1.tiff")

itd.setImg(5000, 5000, 25, 25)
itd.setP(1, 0)
print(list(itd.getMethodResult("Degree of Circular Polarization")))