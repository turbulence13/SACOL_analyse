from __future__ import division
import math
from pyhdf import HDF
from pyhdf.SD import *
import numpy as np
import pandas as pd
import re
import scipy as sp
import scipy.signal as sig
from pyhdf.VS import *
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class hdf_proccess:
    # 设定坐标轴线
    def __init__(self,date):
        self.date = date[0:1]+'-'+date[2:3]+'-'+date[4:5]


    # 两经纬度之间距离计算(单位KM)
    def LonLat_Distance(lonlat1, lonlat2):
        r_earth = 6378.2064
        d_lonlat = math.acos((math.sin(lonlat1[0] * math.pi / 180) * math.sin(lonlat2[0] * math.pi / 180)) +
                             (math.cos(lonlat1[0] * math.pi / 180) * math.cos(lonlat2[0] * math.pi / 180) *
                              math.cos(lonlat1[1] * math.pi / 180 - lonlat2[1] * math.pi / 180))) * r_earth
        return d_lonlat


    # 三点平滑
    def Smooth3(data):
        smoothed_data = data

        for i in range(len(data) - 1):
            smoothed_data[i] = (2 * data[i] + 8 * data[i - 1] + 8 * data[i + 1]) / 18

        smoothed_data[0] = data[0]
        smoothed_data[-1] = data[-1]
        data = smoothed_data
        return smoothed_data


    # 两同形状数组内对应元素遍历相除
    def multi_2(x, y):
        e_a = np.true_divide(x.values, y.values)
        e_a[y.values < 0.0003] = np.nan
        e_a[y.values < 0] = np.nan
        e = pd.DataFrame(e_a)
        return e


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


    # 五点平滑
    def mean5_3(Series, m):
        n = len(Series)
        a = Series
        b = Series.copy()
        for i in range(m):
            b[0] = (69 * a[0] + 4 * (a[1] + a[3]) - 6 * a[2] - a[4]) / 70
            b[1] = (2 * (a[0] + a[4]) + 27 * a[1] + 12 * a[2] - 8 * a[3]) / 35
            for j in range(2, n - 2):
                b[j] = (-3 * (a[j - 2] + a[j + 2]) + 12 * (a[j - 1] + a[j + 1]) + 17 * a[j]) / 35
            b[n - 2] = (2 * (a[n - 1] + a[n - 5]) + 27 * a[n - 2] + 12 * a[n - 3] - 8 * a[n - 4]) / 35
            b[n - 1] = (69 * a[n - 1] + 4 * (a[n - 2] + a[n - 4]) - 6 * a[n - 3] - a[n - 5]) / 70
            a = b.copy()
        return a


    def HDF_route_get(path):
        st_obj = SD(path, SDC.READ)
        Lats = st_obj.select('Latitude').get().T
        Lons = st_obj.select('Longitude').get().T
        st_obj.end()
        L_route = np.concatenate([Lats, Lons]).T
        return L_route


    # 生成范围内文件列表
    def Files_in_range(path_l1, path_vfm=None):
        l1_pathlist = os.listdir(path_l1)

        f_list = []
        for fname in l1_pathlist:
            tem_name = re.match('^CAL_LID_L1-Standard-V4-10\..*\.hdf$', fname)
            if tem_name is not None:
                f_list.append(fname)
        f_mindis = []
        f_count = []
        f_in_range = []
        f_in_date = []
        f_in_time = []
        vfm_in_range = []
        for files in f_list:
            if path_vfm is not None:
                vfm_name = path_vfm + 'CAL_LID_L2_VFM-Standard-V4-20.' + files[-32:]
                if os.path.exists(vfm_name):
                    L_route = HDF_route_get(vfm_name)
                    count = 0
                    min_distance = 999999.0
                    for location in L_route:
                        distance = LonLat_Distance(location, LZU_LatLon)
                        if min_distance > distance:
                            min_distance = distance
                        if distance < 50:
                            count += 1

                    if count > 15:
                        if path_vfm is not None:
                            if os.path.exists(vfm_name):
                                f_mindis.append(min_distance)
                                f_count.append(count)
                                f_in_range.append(files)
                                f_in_date.append(files[-32:-22])
                                f_in_time.append(files[-21:-13])
                                vfm_in_range.append(vfm_name)
                else:
                    print('CAL_LID_L2_VFM-Standard-V4-20.' + files[-32:] + ' VFM Not Found')

            else:
                L_route = HDF_route_get(files)
                min_distance = 999999.0
                count = 0

                for location in L_route:
                    distance = LonLat_Distance(location, LZU_LatLon)
                    if min_distance > distance:
                        min_distance = distance
                    if distance < 50:
                        count += 1

                if count > 15:
                    f_mindis.append(min_distance)
                    f_count.append(count)
                    f_in_range.append(files)
                    f_in_date.append(files[-32:-22])
                    f_in_time.append(files[-21:-13])

        fileslist = {
            'files name': f_in_range,
            'files date': f_in_date,
            'files time': f_in_time,
        }
        if path_vfm is not None:
            fileslist['vfm name'] = vfm_in_range
        f1_info = pd.DataFrame(data=fileslist)
        f1_info.to_csv('selected files.csv')


    def min_is_zero(x):
        x_array = x.values
        x_array[x_array < 0] = 0
        y = pd.DataFrame(x_array)
        return y


    # 生成高度范围内数据集字典
    def Data_dic_select(dic, _min, _max):
        l_Rd_dic = {}
        for key in dic:
            l_Rd_dic[key] = dic[key].loc[:, dic[key].columns < _max]
            l_Rd_dic[key] = l_Rd_dic[key].loc[:, l_Rd_dic[key].columns > _min]
        return l_Rd_dic


    def L2_VFM_Reading(fpath):

        sd_obj = SD(fpath, SDC.READ)
        Vt_obj = HDF.HDF(fpath).vstart()
        m_data = Vt_obj.attach('metadata').read()[0]
        Height = np.array(m_data[-1])  # 583高度对应实际海拔
        Lats = sd_obj.select('Latitude').get()
        Lons = sd_obj.select('Longitude').get()
        L_route = np.concatenate([Lats.T, Lons.T]).T
        target_rows = []

        for location in L_route:
            distance = LonLat_Distance(location, LZU_LatLon)
            if distance < 50:
                target_rows.append(True)
            else:
                target_rows.append(False)

        VFM_basic = np.array(sd_obj.select('Feature_Classification_Flags').get())
        VFM_basic = VFM_basic % 8
        VFM_1 = np.reshape(VFM_basic[:, 0:165], (VFM_basic.shape[0] * 3, 55))
        VFM_1 = np.repeat(VFM_1, 5, axis=0)
        VFM_2 = np.reshape(VFM_basic[:, 165:1165], (VFM_basic.shape[0] * 5, 200))
        VFM_2 = np.repeat(VFM_2, 3, axis=0)
        VFM_3 = np.reshape(VFM_basic[:, 1165:5515], (VFM_basic.shape[0] * 15, 290))
        VFM = np.concatenate((VFM_1, VFM_2, VFM_3), axis=1)
        target_rows_VFM = np.repeat(target_rows, 15)
        Rd_dic = {}
        Rd_dic['VFM'] = VFM
        Rd_dic_meta = {
            'route': L_route,
            'Lats': Lats,
            'target rows': target_rows,
            'Height': Height,
            'target rows VFM': target_rows_VFM,
        }
        sd_obj.end()
        HDF.HDF(fpath).close()
        return Rd_dic, Rd_dic_meta


    # 获取文件内数据字典
    def L1_Reading(fpath):
        sd_obj = SD(fpath, SDC.READ)
        Vt_obj = HDF.HDF(fpath).vstart()
        m_data = Vt_obj.attach('metadata').read()[0]
        Height = np.array(m_data[-2])  # 583高度对应实际海拔
        Lats = sd_obj.select('Latitude').get()
        Lons = sd_obj.select('Longitude').get()
        L_route = np.concatenate([Lats.T, Lons.T]).T
        target_rows = []

        for location in L_route:
            distance = LonLat_Distance(location, LZU_LatLon)
            if distance < 50:
                target_rows.append(True)
            else:
                target_rows.append(False)

        Per532 = np.array(sd_obj.select('Perpendicular_Attenuated_Backscatter_532').get())
        Per532 = sig.medfilt(Per532, [1, 15])
        Per532[Per532 < 0] = 0
        Tol532 = np.array(sd_obj.select('Total_Attenuated_Backscatter_532').get())
        Tol532 = sig.medfilt(Tol532, [1, 15])
        Tol532[Tol532 < 0] = 0
        Par532 = Tol532 - Per532
        Par532 = sig.medfilt(Par532, [1, 15])
        # proccess Dep data

        Dep532 = np.true_divide(Per532, Par532)
        Dep532[Par532 <= 0.0003] = 0
        Dep532[Par532 <= 0.0000] = 0
        Dep532[Dep532 > 1] = 1
        Data_dic = {}
        Data_dic['Tol532'] = Tol532
        # Rd_dic['Per532'] = Per532
        # Rd_dic['Par532'] = Par532
        Data_dic['Dep532'] = Dep532
        Data_meta = {
            'route': L_route,
            'Lats': Lats,
            'target rows': target_rows,
            'Height': Height,
        }
        # for key, value in Rd_dic.items():
        # value.columns = Height.values[0]
        sd_obj.end()
        HDF.HDF(fpath).close()
        return Data_dic, Data_meta


    def clr_dep(dep):
        depc = dep.copy()
        depc[depc > 1] = 1.0
        depc = depc * 10
        idep = np.trunc(depc) + 1
        idep[(depc < 0.2) & (depc >= 0.06)] = 2
        idep[depc is np.nan] = 0
        return idep


    path1 = 'E:/Files Data/SACOL/L1_data/'
    path_vfm = 'E:/Files Data/SACOL/VFM_data/'
    print(os.path.exists(path_vfm))
    pic_folder = 'Fig/'
    os.chdir(path1)
    try:  # 文件夹创建，用于保存图片，若存在则在不创建
        os.mkdir(path=path1 + pic_folder)
    except FileExistsError:
        print(' folder exist')

    try:  # 文件夹创建，用于保存图片，若存在则在不创建
        os.mkdir(path=path1 + pic_folder+'heat/')
    except FileExistsError:
        print(' folder exist')

    try:  # 文件夹创建，用于保存图片，若存在则在不创建
        os.mkdir(path=path1 + pic_folder+'by_height/')
    except FileExistsError:
        print(' folder exist')


    LZU_LatLon = [35.946, 104.137]
    Files_in_range(path1, path_vfm=path_vfm)  # 执行文件筛选,会输出存储文件，只执行一次便于其他部分调试
    target_flist = pd.read_csv('selected files.csv', sep=',', header=0)  # 筛选结果读取

    for d_t, row in target_flist.iterrows():
        f_path = path1 + '/' + row["files name"]
        L1_dic, L1_meta = L1_Reading(f_path)
        vfm_path = row["vfm name"]
        VFM_dic, VFM_meta = L2_VFM_Reading(vfm_path)
        L1_frame_dic = {}
        clear_L1_Data = {}
        # Rd_dic['Dep532'] = clr_dep(Rd_dic['Dep532'])
        target_VFM = {}
        target_L1 = {}

        target_route = VFM_meta['route'][VFM_meta['target rows']]
        if target_route[0][0] < target_route[-1][0]:
            loc_range = [target_route[0][0], target_route[-1][0]]
        else:
            loc_range = [target_route[-1][0], target_route[0][0]]
        ttt = (loc_range[0] <= L1_meta['Lats']) & (loc_range[1] >= L1_meta['Lats'])
        fff = ttt.copy()
        for i in range(len(ttt)):
            if ttt[i][0]:
                fff[i+14][0] = True
        '''    
        print(len(fff))
        print(L1_meta['route'][fff.T[0]].shape)
        print(target_route.shape)
        '''

        for key in L1_dic:
            target_L1[key] = L1_dic[key][fff.T[0]]
        for key in VFM_dic:
            target_VFM[key] = VFM_dic[key][VFM_meta['target rows VFM']]
        cloud_status = []

        for i in target_VFM['VFM']:
            if 2 in target_VFM['VFM'][i]:
                cloud_status.append(False)
            else:
                cloud_status.append(True)

        if cloud_status.count(True)/len(cloud_status) >= 0.0:
            for keys in target_L1:
                L1_frame_dic[keys] = pd.DataFrame(target_L1[keys])
                L1_frame_dic[keys].columns = L1_meta['Height']
                clear_L1_Data[keys] = pd.DataFrame(target_L1[keys][cloud_status])
                clear_L1_Data[keys].columns = L1_meta['Height']

            l_Rd_dic = Data_dic_select(L1_frame_dic, -0.5, 30.1)
            clear_L1_Dic = Data_dic_select(clear_L1_Data, 0, 30.1)
            Avg_Rd = {}
            for keys in clear_L1_Data:
                Avg_Rd[keys] = np.nanmean(clear_L1_Dic[keys].values, axis=0)
                Avg_Rd[keys] = mean5_3(Avg_Rd[keys], 10)
            fig_path = path1 + pic_folder + 'by_height/' + row["files date"]+'_'+row["files time"]
            heat_path = path1 + pic_folder + 'heat/' + row["files date"]+'-'+row["files time"]+'_heat'
            # plt.savefig(fig_path)
            plt.plot(Avg_Rd['Dep532'], clear_L1_Dic['Dep532'].columns)
            plt.savefig(fig_path)
            plt.close()

            f, ax = plt.subplots(nrows=2, figsize=(8, 6))
            sns.heatmap(l_Rd_dic['Dep532'].T, vmin=0, vmax=1.0, cmap='depratio', ax=ax[0])
            sns.heatmap(target_VFM['VFM'].T, vmax=7, cmap='depratio', ax=ax[1])
            allspines_set(ax[0])
            allspines_set(ax[1])
            plt.savefig(heat_path)
            plt.close()

    # plt.plot(Avg_Rd['Par532'], l_Height)
    # plt.plot(Avg_Rd['Dep532'], l_Height)
''' 
        f, ax = plt.subplots(nrows=2, figsize=(8, 6))
        sns.heatmap(l_Rd_dic['Tol532'].T, vmin=0, vmax=0.005, cmap='depratio', ax=ax[0])
        sns.heatmap(target_VFM['VFM'].T, vmax=7, cmap='depratio', ax=ax[1])
        allspines_set(ax[0])
        allspines_set(ax[1])
'''

'''

        It532 = pd.DataFrame(st_obj.select('Perpendicular_Attenuated_Backscatter_532').get())
        Dp532 = pd.DataFrame(st_obj.select('Total_Attenuated_Backscatter_532').get())
        It1064 = pd.DataFrame(st_obj.select('Attenuated_Backscatter_1064').get())
        Rd_dic = {'It532': It532, 'Dp532': Dp532, 'It1064': It1064}

        for key, value in Rd_dic.items():
            value.columns = Height.values[0]
            value.index = Lons.values[0]
        # Rdframe = pd.DataFrame(It532,index=Lats,columns=Height)
        # f, ax = plt.subplots(nrows=1, figsize=(10, 5))
        # sb.heatmap(data=Rd_dic['Dp532'].T, cmap='customcb', vmax=0.4, vmin=0, ax=ax)  # 采用ax解决坐标轴问题
        # plt.show()
        st_obj.end()
        sys.exit()
        # print(Heights)

            Data_atr = st_obj.attributes()
            Data_dic = st_obj.datasets().keys()

# color setting
clrlist = []

colorbar.ColorbarBase

fig = plt.figure()

plt.plot(It532)

plt.show()
print(It1064)

'''
