



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
        pass

    def getMean(self, ):
        pass

    def getMAD(self):
        pass

    def getContrast(self):
        pass

    def getCorrelation(self):
        pass

    def getEntropy(self):
        pass

    def getASM(self):
        pass

    def getIDM(self):
        pass