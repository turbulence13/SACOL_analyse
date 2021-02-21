import numpy as np
import pandas as pd
import os


def date_files_reading(date, path):
    files = ('SACOL_NIESLIDAR_' + date + '_Int532_Dep532_Int1064.dat')
    os.chdir(path)
    f_data = pd.read_table(files, sep='\s+', index_col='Height(km)', na_values=['NaN'], skiprows=3)
    data = {
        'It532': f_data.iloc[0:3000][:],
        'Dp532': f_data.iloc[3000:6000][:],
    }
    return data


def coefficient_read():
    path_after = 'E:/Files Data/SACOL/2020.04.27-Depolarization Ratio Correction Test/' \
                 '2020.04.27-Dep Ratio-Correction start- 2020.04.27-within 1-1.5km mean.csv'
    path_before = 'E:/Files Data/SACOL/2020.04.27-Depolarization Ratio Correction Test/' \
                  '2020.04.27-Dep Ratio-Correction start- 2020.04.27-within 1-1.5km mean.csv'

    before_data = pd.read_csv(path_before, sep=',', index_col='Height', na_values=['NaN'], skiprows=2,
                              usecols=['Height', 'Ch2/Ch1_00', 'Ch2/Ch1_10'])
    after_data = pd.read_csv(path_after, sep=',', index_col='Height', na_values=['NaN'], skiprows=2,
                             usecols=['Height', 'Ch2/Ch1_00', 'Ch2/Ch1_10'])

    return before_data, after_data
