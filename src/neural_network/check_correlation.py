import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import seaborn as sns

def reduce_mem_usage(df : pd.DataFrame):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo('f2').min and c_max < np.finfo('f2').max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo('f4').min and c_max < np.finfo('f4').max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo('i1').min and c_max < np.iinfo('i1').max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo('i2').min and c_max < np.iinfo('i2').max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo('i4').min and c_max < np.iinfo('i4').max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo('i8').min and c_max < np.iinfo('i8').max:
                df[col] = df[col].astype(np.int64)
        elif str(col_type)[:8] == 'datetime':
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Потребление памяти меньше на',
         round(start_mem - end_mem, 2),
         'Мб (минус',
         round(100 * (start_mem - end_mem) / start_mem, 1),
         '%)')
    return df


methods = {
    "SHG Intensity" : [
        "SHG Intensity Mean",
        "SHG Intensity MAD",
        "SHG Intensity Contrast",
        "SHG Intensity Correlation",
        "SHG Intensity Entropy",
        "SHG Intensity ASM",
        "SHG Intensity IDM"
    ],
    "R-Ratio" : [
        "R-Ratio Mean",
        "R-Ratio MAD",
        "R-Ratio Contrast",
        "R-Ratio Correlation",
        "R-Ratio Entropy",
        "R-Ratio ASM"
    ],
    "Degree of Circular Polarization" : [
        "Degree of Circular Polarization Mean",
        "Degree of Circular Polarization MAD",
        "Degree of Circular Polarization Contrast",
        "Degree of Circular Polarization Correlation",
        "Degree of Circular Polarization Entropy",
        "Degree of Circular Polarization ASM",
        "Degree of Circular Polarization IDM"
    ],
    "SHG-CD" : [ 
        "SHG-CD MAD",
        "SHG-CD Contrast",
        "SHG-CD Correlation",
        "SHG-CD Entropy",
        "SHG-CD ASM",
        "SHG-CD IDM"
    ],
    "SHG-LD" : [
        "SHG-LD MAD",
        "SHG-LD Contrast",
        "SHG-LD Correlation",
        "SHG-LD Entropy",
        "SHG-LD ASM",
        "SHG-LD IDM"
    ],
    "Params" : [
        "2-Group Tag",
        "Pixel Density",
    ]
}

x_axis = sum([methods[key] for key in methods.keys() if key != "Params"], [])
# x_axis += [methods['Params'][1]]
y_axis = methods["Params"][0]
def getData(table_number):
    data = pd.read_excel(io="../../Data/41598_2022_13623_MOESM3_ESM.xlsx", 
    sheet_name=f"{1 << 2 * (table_number - 1)} Subimage Training")
    data = reduce_mem_usage(data)
    data = (data - data.min()) / (data.max() - data.min())
    #data["2-Group Tag"] = data[y_axis] == 2
    return data

data = getData(4)
sns.pairplot(data, hue=y_axis, x_vars=methods["Degree of Circular Polarization"], y_vars=methods["SHG-LD"])

plt.show()