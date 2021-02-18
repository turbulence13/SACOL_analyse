import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import math
from pyhdf.SD import *
from pyhdf import HDF
from pyhdf.VS import *
import scipy.signal as sig
import re
import hdf_reading as hr
import mean_proccess as mp


def allspines_set(ax, is_on=True, width=1):  # 坐标轴线格式
    if is_on:
        for spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(width)
    else:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)


def Radar_heat(data_dic, time_area=None, height_area=None):  # 针对本次绘图设计绘图函数，应当针对性进行改变
    x_minorlocator = AutoMinorLocator(n=3)
    y_ticks = np.linspace(0, 1677, 6)
    y_label = ('0.0', '2.0', '4.0', '6.0', '8.0', '10.0')
    f, ax = plt.subplots(nrows=len(data_dic), sharex=True, figsize=(6, 6))
    i = 0
    ax_dic = {}
    for key in data_dic:
        ax_dic[key] = ax[i]
        i = i + 1
    sns.heatmap(data_dic['It532'], vmax=40.0, vmin=0.0, cmap='customcb',
                ax=ax_dic['It532'], yticklabels=400, xticklabels=18)
    ax_dic['It532'].invert_yaxis()
    ax_dic['It532'].set_xticks(np.linspace(0, 1440, 8))
    sns.heatmap(data_dic['Dp532'], vmax=0.3, vmin=0.0, cmap='customcb',
                ax=ax_dic['Dp532'], yticklabels=400, xticklabels=18)
    ax_dic['Dp532'].invert_yaxis()
    ax_dic['Dp532'].set_xlabel('Time')
    if (time_area is not None) & (height_area is not None):
        height_c = height_area.copy()
        height_c[0] = height_c[0]*166.6666
        height_c[1] = height_c[1]*166.6666
        for keys in ax_dic:  # 坐标轴刻度格式
            ax_dic[keys].vlines(time_area, ymin=height_c[0], ymax=height_c[1], colors='black',
                                linestyles='dashed')
            ax_dic[keys].hlines(height_c, xmin=time_area[0], xmax=time_area[1], colors='black',
                                linestyles='dashed')
    for keys in ax_dic:  # 坐标轴刻度格式
        ax_dic[keys].set_yticks(y_ticks)
        ax_dic[keys].set_yticklabels(y_label, rotation=0)
        ax_dic[keys].minorticks_on()
        ax_dic[keys].xaxis.set_minor_locator(x_minorlocator)
        allspines_set(ax_dic[keys], width=1)  # 坐标轴框线


def dep_by_height(data, meantime=1, top=10.0, bottum=0.0):
    data_a = data.copy()
    # data_a[np.isnan(data_a)] = 0
    # data_a[np.isinf(data_a)] = 0
    data_b = np.nanmean(data_a, axis=1)
    data_b[data_b < 0] = 0
    data_b = mp.mean_simple(data_b, meantime)
    data_c = pd.DataFrame(data=data_b, index=data.index)
    avg_data = np.nanmean(data_c.loc[(data_c.index <= top) & (data_c.index >= bottum)].values)
    return data_c, avg_data


def plot_by_height(series, top=10.0, bottum=0.0, horizontal=None):
    if horizontal is None:
        horizontal = [0, 0.1]
    plt.figure(figsize=(3, 4.5))
    plt.axis([horizontal[0], horizontal[1], top, bottum])
    plt.plot(series.values, series.index, color='black', linewidth=1.0)
    # fig.xticks(np.linspace(0, 1440, 8))


def date_files_reading(date, path):
    files = ('SACOL_NIESLIDAR_' + date + '_Int532_Dep532_Int1064.dat')
    os.chdir(path)
    f_data = pd.read_table(files, sep='\s+', index_col='Height(km)', na_values=['NaN'], skiprows=3)
    data = {
        'It532': f_data.iloc[0:3000][:],
        'Dp532': f_data.iloc[3000:6000][:],
    }
    return data


def date_L1_reading(date, path, path_vfm):

    os.chdir(path)
    f_list = os.listdir(path)
    t_date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
    for files in f_list:
        fname = re.match('^CAL_LID_L1-Standard-V4-10\.'+t_date+'.*\.hdf$',files)
        if fname is not None:
            vfm_files = path_vfm + 'CAL_LID_L2_VFM-Standard-V4-20' + files[25:]
            Dp_height_clear, Data_height = hr.L1_VFM_proccess(files, vfm_files)
            # Data_mean = np.nanmean(Data_dic['Dep532'], axis=0)
            height = Data_height - 1.961
            Data_height = pd.DataFrame(Dp_height_clear, index=height)

    return Data_height


def target_average_dp(date, path, time_area, height_area):

    # 文件读取，跳过文件说明，选取高度作为行名，便于画图
    Rddata_dic = date_files_reading(date, path)
    Rddata_dic['Dp532'].values[Rddata_dic['Dp532'].values < 0] = np.nan
    Rddata_dic['Dp532'].values[Rddata_dic['Dp532'].values > 1] = np.nan
    Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                       meantime=3, top=height_area[1], bottum=height_area[0])
    return avgdata, Dp_height


def Satellite_compare(date, path_SACOL, path_L1, path_vfm, path_f, time_area=None,
                      height_area=[0, 10], calibration=None, horizontal=[0.0, 0.4]):

    if not os.path.exists(path_f + '/dep_height/'):
        os.mkdir(path=path_f + '/dep_height/')
    f_path = path_f + '/dep_height/' + date
    path_L1 = path_L1
    Sacol_data = date_files_reading(date, path_SACOL)
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values < 0] = np.nan
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values > 1] = np.nan
    L1_data = date_L1_reading(date, path_L1, path_vfm)
    Dp_height, avgdata = dep_by_height(Sacol_data['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                       meantime=3, top=height_area[1], bottum=height_area[0])
    aaa = str(avgdata)[:10]
    if calibration is not None:
        cal_Dp = Dp_height - calibration
        plot_by_height(cal_Dp, top=height_area[0], bottum=height_area[1], horizontal=horizontal)
    else:
        plot_by_height(Dp_height, top=height_area[0], bottum=height_area[1], horizontal=horizontal)

    plt.plot(L1_data.values, L1_data.index, color='red', linewidth=1.0)
    plt.savefig(f_path)
    plt.close()


def Calibrate_procces(date, path, pathf, time_area=None, height_area=None, calibration=None, horizontal=[0.0, 0.4]):

    if not os.path.exists(pathf + '/dep_height/'):
        os.mkdir(path=pathf + '/dep_height/')
    if not os.path.exists(pathf + '/heat_map/'):
        os.mkdir(path=pathf + '/heat_map/')
    f_path = pathf + '/dep_height/' + date
    f_path_heat = pathf + '/heat_map/' + date  # 根据文件创立图像文件夹(可优化)

    # 文件读取，跳过文件说明，选取高度作为行名，便于画图
    Sacol_data = date_files_reading(date, path)
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values < 0] = np.nan
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values > 1] = np.nan
    l_Rdd_dic = {}

    for keys in Sacol_data:
        l_Rdd_dic[keys] = Sacol_data[keys].loc[(Sacol_data[keys].index < 10) & (Sacol_data[keys].index > 0)]
    if calibration is not None:
        l_Rdd_dic['Dp532'] = l_Rdd_dic['Dp532'] - calibration

    if (time_area is not None) & (height_area is not None):
        Dp_height, avgdata = dep_by_height(Sacol_data['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                           meantime=3, top=height_area[1], bottum=height_area[0])
        aaa = str(avgdata)
        aaa = aaa[:10]
        plot_by_height(Dp_height, top=height_area[0], bottum=height_area[1], horizontal=horizontal)

        if calibration is not None:
            cal_Dp = Dp_height - calibration
            plt.plot(cal_Dp.values, cal_Dp.index, color='black', linewidth=1.0)

        plt.text(x=0.03, y=np.mean(height_area), s='Avg:\n'+aaa)
        plt.savefig(f_path)
        plt.close()

    Radar_heat(l_Rdd_dic, time_area, height_area)
    plt.savefig(f_path_heat)
    plt.close()


# pathf = input('Target Folder Path:')
path1 = 'E:/Files Data/SACOL/NIESdat'  # 目标文件夹路径
pathfig = 'E:/Files Data/SACOL/Figure/'
path_L1 = 'E:/Files Data/SACOL/L1_data/'
path_vfm = 'E:/Files Data/SACOL/VFM_data/'

LZU_LatLon = [35.946, 104.137]

if not os.path.exists(pathfig):# 文件夹创建，用于保存图片，若存在则在不创建
    os.mkdir(path=pathfig)

cal_dic1 = {
    '20181217': [[120, 132], [2, 2.5]],
    '20190116': [[108, 120], [2, 3]],
    '20190704': [[6, 18], [2.5, 3]],
    '20190710': [[12, 24], [1.5, 2]],
}
cal_dic2 = {
    '20191026': [[84, 96], [2.5, 3]],
    '20191028': [[36, 48], [2.5, 3.5]],
}
cal_dic3 = {
    '20191113': [[120, 132], [2, 2.5]],
    '20191209': [[6, 18], [2.5, 3]],
    '20191222': [[6, 18], [2, 2.5]],
    '20200309': [[72, 84], [1.8, 2.2]]
}
cal_dic4 = {
    '20200430': [[108, 120], [4.2, 5.5]],
    '20200501': [[18, 30], [3.9, 4.3]],
}
cal_dic5 = {
    '20200617': [[84, 96], [3.4, 4]],
    '20200801': [[90, 102], [3.1, 3.5]],
    '20200918': [[54, 66], [3.0, 3.5]],
}
cal_main_dic = {
    '1': cal_dic1,
    '2': cal_dic2,
    '3': cal_dic3,
    '4': cal_dic4,
    '5': cal_dic5,
}
cal_dic = {}
process_list = ['1', '2', '3', '4', '5']

satel_dic1 = {
    '20181116': [[111, 117], [0, 10]],
    '20190112': [[111, 117], [0, 10]],
    '20190501': [[112, 118], [0, 10]],
}
satel_dic2 = {
    '20191113': [[33, 39], [0, 10]],
    '20191222': [[33, 39], [0, 10]],
}
satel_dic3 = {
    '20191113': [[33, 39], [0, 10]],
    '20191222': [[33, 39], [0, 10]],
    '20200104': [[33, 39], [0, 10]],
}
satel_dic4 = {
    '20200430': [[34, 40], [0, 10]],
}
satel_dic5 = {
    '20200625': [[114, 120], [0, 10]],
}

satel_main_dic = {
    '1': satel_dic1,
    '2': satel_dic2,
    '3': satel_dic3,
    '4': satel_dic4,
    '5': satel_dic5,
}

'''
os.chdir(path1)
all_file_list = os.listdir()
for file in all_file_list:
    if file[-4:] == '.dat':
        date = file[16:24]
        Main_procces(date, path1, pathfig+'ALL')
'''

for num in process_list:
    path_plot_dir = pathfig+num
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)
    cal_list = []
    for key in cal_main_dic[num]:
        Calibrate_procces(key, path1, path_plot_dir, time_area=cal_main_dic[num][key][0],
                     height_area=cal_main_dic[num][key][1], horizontal=[0, 0.02])
        avg_dp, dp_height = target_average_dp(key, path1, time_area=cal_main_dic[num][key][0],
                                              height_area=cal_main_dic[num][key][1])
        cal_list.append(avg_dp-0.0044)
    cal_dic[num] = np.min(cal_list)

    path_plot_dir = pathfig + num +'_all_height'
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)

    for key in cal_main_dic[num]:
        Calibrate_procces(key, path1, path_plot_dir, time_area=cal_main_dic[num][key][0],
                          height_area=[0, 5], calibration=cal_dic[num], horizontal=[0, 0.1])

    path_plot_dir = pathfig + num +'_satellite'
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)

    for key in satel_main_dic[num]:
        Satellite_compare(key, path1, path_L1, path_vfm, path_plot_dir, time_area=satel_main_dic[num][key][0],
                          height_area=[0, 15], calibration=cal_dic[num], horizontal=[0.0, 0.4])



'''
Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].loc['12:00':'17:00'], meantime=1)
print(avgdata)
    
plot_by_height(Dp_height)

plt.savefig(f_path)
plt.close()

l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values < 0] = 0
l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values > 1] = 1
print(l_Rdd_dic['Dp532'])
plot_by_height(l_Rdd_dic['Dp532'].iloc[:, 80])'''

'''
        sns.heatmap(Rddata_dic['It1064'], vmax=1.0, vmin=0.0, cmap='rainbow', ax=ax_dic['It1064'])
        ax_dic['It1064'].invert_yaxis()
        ax_dic['It1064'].set_yticks(np.linspace(0, 6, 4))
        ax_dic['It1064'].set_xlabel('Time')

        for keys in Rddata_dic:
            sns.set_style()

            sns.heatmap(Rddata_dic[keys], vmax=0.4, vmin=0.0, cmap='rainbow', ax=ax_dic[keys])
            ax_dic[keys].invert_yaxis()
            ax_dic[keys].set_yticks([0.0, 6.0])
            ax_dic[keys].set_xlabel('Time')

            print(keys)
            #plt.savefig(f_name+'_'+keys+'.png', dpi=300)
            plt.show()
            
        x_range = np.linspace(0, 255, 256)
        r_f = np.interp(x_range, [0, 1, 50, 150, 200, 255], [255, 255, 0, 10, 255, 255])
        g_f = np.interp(x_range, [0, 1, 50, 100, 200, 255], [255, 255, 0, 255, 255, 0])
        b_f = np.interp(x_range, [0, 100, 150, 255], [255, 255, 10, 0])
        cus_rgb = (np.concatenate([[r_f], [g_f], [b_f]]).T)/255

        print(cus_rgb)
'''
