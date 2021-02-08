import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns


def mean5_3(Series, m):
    n = len(Series)
    a = Series
    b = Series.copy()
    print(b)
    for i in range(m):
        b[0] = (69 * a[0] + 4 * (a[1] + a[3]) - 6 * a[2] - a[4]) / 70
        b[1] = (2 * (a[0] + a[4]) + 27 * a[1] + 12 * a[2] - 8 * a[3]) / 35
        for j in range(2, n - 2):
            b[j] = (-3 * (a[j - 2] + a[j + 2]) + 12 * (a[j - 1] + a[j + 1]) + 17 * a[j]) / 35
        b[n - 2] = (2 * (a[n - 1] + a[n - 5]) + 27 * a[n - 2] + 12 * a[n - 3] - 8 * a[n - 4]) / 35
        b[n - 1] = (69 * a[n - 1] + 4 * (a[n - 2] + a[n - 4]) - 6 * a[n - 3] - a[n - 5]) / 70
        a = b.copy()
    return a


def mean_simple(Series, m):
    n = len(Series)
    a = Series
    b = Series.copy()
    for i in range(m):
        b[0] = (a[0] + a[1]) / 2
        for j in range(1, n - 1):
            b[j] = (a[j - 1] + a[j + 1] + a[j]) / 3
        b[n - 1] = (a[n - 1] + a[n - 2]) / 2
        a = b.copy()
    return a


def allspines_set(ax, is_on=True, width=1):  # 坐标轴线格式
    if is_on:
        for spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(width)
    else:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)


# def Dfselect_inrange(data,imax,imin):
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
        if (time_area is not None) & (height_area is not None):
            ax_dic[keys].vlines(time_area, ymin=height_c[0], ymax=height_c[1], colors='black',
                                linestyles='dashed')
            ax_dic[keys].hlines(height_c, xmin=time_area[0], xmax=time_area[1], colors='black',
                                linestyles='dashed')
        ax_dic[keys].set_yticks(y_ticks)
        ax_dic[keys].set_yticklabels(y_label, rotation=0)
        ax_dic[keys].minorticks_on()
        ax_dic[keys].xaxis.set_minor_locator(x_minorlocator)
        allspines_set(ax_dic[keys], width=1)  # 坐标轴框线


def dep_by_height(data, meantime=1, top=10.0, bottum=0.0):
    data_a = data.copy()
    data_a[np.isnan(data_a)] = 0
    data_a[np.isinf(data_a)] = 0
    data_b = np.nanmean(data_a, 1)
    data_b[data_b < 0] = 0
    data_b = mean_simple(data_b, meantime)
    data_c = pd.DataFrame(data=data_b, index=data.index)

    avg_data = np.mean(data_c.loc[(data_c.index <= top) & (data_c.index >= bottum)].values)
    return data_c, avg_data


def plot_by_height(series, top=10.0, bottum=0.0):
    plt.figure(figsize=(3, 4.5))
    plt.axis([0, 0.05, bottum, top])
    x_minorlocator = AutoMinorLocator(n=3)
    plt.plot(series.values, series.index, color='black', linewidth=1.0)
    # fig.xticks(np.linspace(0, 1440, 8))


def Main_procces(date, path, pathf, time_area=None, height_area=None):
    files = ('SACOL_NIESLIDAR_' + date + '_Int532_Dep532_Int1064.dat')
    os.chdir(path)
    try:  # 文件夹创建，用于保存图片，若存在则在不创建
        os.mkdir(path=pathf + '/dep_height/')
    except FileExistsError:
        print('fig folder exist')
    try:
        os.mkdir(path=pathf + '/heat_map/')
    except FileExistsError:
        print('heat folder exist')

    f_path = pathf + '/dep_height/' + date
    f_path_heat = pathf + '/heat_map/' + date  # 根据文件创立图像文件夹(可优化)
    # 文件读取，跳过文件说明，选取高度作为行名，便于画图
    f_data = pd.read_table(files, sep='\s+', index_col='Height(km)', na_values=['NaN'], skiprows=3)
    Rddata_dic = {
        'It532': f_data.iloc[0:3000][:],
        'Dp532': f_data.iloc[3000:6000][:],
    }
    l_Rdd_dic = {}

    for keys in Rddata_dic:
        l_Rdd_dic[keys] = Rddata_dic[keys].loc[(Rddata_dic[keys].index < 10) & (Rddata_dic[keys].index > 0)]
    l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values < 0] = np.nan
    l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values > 1] = np.nan

    if (time_area is not None) & (height_area is not None):
        Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                           meantime=3, top=height_area[1], bottum=height_area[0])
        aaa = str(avgdata)
        aaa = aaa[:10]
        plot_by_height(Dp_height, top=height_area[0], bottum=height_area[1])
        print(np.mean(height_area))
        plt.text(x=0.03, y=np.mean(height_area), s='Avg:\n' + aaa)
        plt.savefig(f_path)
        plt.close()
        print(avgdata)

    Radar_heat(l_Rdd_dic, time_area, height_area)
    plt.savefig(f_path_heat)
    plt.close()



# pathf = input('Target Folder Path:')
path1 = 'E:/Files Data/SACOL/NIESdat'  # 目标文件夹路径
pathfig = 'E:/Files Data/SACOL/NIESdat/Figure/ALL'

try:  # 文件夹创建，用于保存图片，若存在则在不创建
    os.mkdir(path=path1+'/Figure')
except FileExistsError:
    print('fig folder exist')
try:  # 文件夹创建，用于保存图片，若存在则在不创建
    os.mkdir(path=pathfig)
except FileExistsError:
    print('fig folder exist')

files_dic1 = {
    '20181217': [[120, 132], [4, 5.5]],
    '20190116': [[108, 120], [4, 5.5]],
    '20190704': [[6, 18], [4, 5.5]],
    '20190710': [[6, 18], [3, 4.5]]
}

files_dic3 = {
    '20191113': [[120, 132], [3, 4.5]],
    '20191209': [[6, 18], [2.5, 4]],
    '20191222': [[6, 18], [2, 4]],
    '20200309': [[72, 84], [2, 5]]
}

files_dic2 = {
    '20181217': [[0, 144], [4, 5.5]],
    '20190116': [[90, 144], [4, 5.5]],
    '20190704': [[0, 54], [4, 5.5]],
    '20190710': [[0, 36], [3, 4.5]]
}
os.chdir(path1)
filelist = os.listdir()
for file in filelist:
    print(file[-4:])
    if file[-4:] == '.dat':
        print(file)
        date_temp = file[16:24]
        Main_procces(date_temp, path1, pathfig)



    '''
    
for key in files_dic3:
    fname = ('SACOL_NIESLIDAR_' + key + '_Int532_Dep532_Int1064.csv')
    Main_procces(key, path1, pathfig, files_dic3[key][0], files_dic3[key][1])
    
    
    Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].loc['12:00':'17:00'], meantime=1)
    print(avgdata)
        
    plot_by_height(Dp_height)

    plt.savefig(f_path)
    plt.close()
    '''

'''        l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values < 0] = 0
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
