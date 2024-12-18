#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    : result_data.py
@Time    : 2024/11/29
@Author  : MajorCai
@Version : 0.1
@Desc    : #TODO
"""
import csv
import re
from stagesepx.reporter import Reporter
import os
import sys

def write_result_to_local(_video_file_name, _result_dict, _classify_result):
    _from_movie_2_picture = "_Test"
    _video_file_name = "Test.mp4"
    # 待写入csv的一行数据
    result_row = []
    # print(re.search(r'\\(.*).mp4', str(i), re.M | re.I).group(1))
    match = re.search(r'\\(.*).mp4', str(_video_file_name), re.M | re.I)
    if match:
        mp4_filename = match.group(1)
    else:
        # 处理没有找到匹配的情况
        mp4_filename = "default_filename"


    # 将结果写本地
    # txt_html_path = _from_movie_2_picture + '/forecast_stable_' + mp4_filename + '/' + mp4_filename
    f = open(mp4_filename + '.txt', 'a+')
    f.write(str(_result_dict).replace(', ', ',\n'))

    # 处理结果
    result_row.append(mp4_filename + '.mp4')

    # --- draw ---
    r = Reporter()
    r.draw(_classify_result, mp4_filename + '.html')

    # 计算结果
    # 用['-3'][0][0]表示用户点击行为
    if '-3' in _result_dict.keys() and len(_result_dict['-3']) > 0:
        search_obj1 = re.search(r'frame_id=(.*) timestamp=(.*)>', str(_result_dict['-3'][0][0]), re.M | re.I)
        # print('完成点击图标的帧数为第 %s 帧，时间点为第 %s 秒' % (search_obj1.group(1), str(search_obj1.group(2))))
        result_row.append(str(search_obj1.group(2)))
        result_row.append(str(search_obj1.group(1)))
    else:
        # print("未找到用户点击完图标的时间点")
        result_row.append('None')
        result_row.append('None')

    # 有时候，用['1'][0][0]表示用户点击行为
    if '1' in _result_dict.keys() and len(_result_dict['1']) > 0:
        search_obj2 = re.search(r'frame_id=(.*) timestamp=(.*)>', str(_result_dict['1'][0][0]), re.M | re.I)
        # print('开始点击的帧数为第 %s 帧，时间点为第 %s 秒' % (search_obj1.group(1), str(search_obj1.group(2))))
        result_row.append(str(search_obj2.group(2)))
        result_row.append(str(search_obj2.group(1)))
    else:
        # print("未找到开始点击的时间点")
        result_row.append('None')
        result_row.append('None')

    # 进入目标页面
    if '2' in _result_dict.keys() and len(_result_dict['4']) > 0:
        search_obj3 = re.search(r'frame_id=(.*) timestamp=(.*)>', str(_result_dict['4'][0][0]), re.M | re.I)
        # print('缓冲结束，进入目标页面的帧数为第 %s 帧，时间点为第 %s 秒' % (search_obj3.group(1), search_obj3.group(2)))
        result_row.append(str(search_obj3.group(2)))
        result_row.append(str(search_obj3.group(1)))
    else:
        # print("未找到进入目标页面的时间点")
        result_row.append('None')
        result_row.append('None')

    return result_row

def process_csv(_actual_result, _forecast_result, _csv_output):
    _actual_result = "actual_result.csv"
    _csv_output = "csv_output.csv"
    # 输入的实际结果的一整行数据，但是每一行的文件名可能和期望结果不一致
    actual_result_row = []
    with open(_actual_result, 'r', encoding='gbk') as f1:
        reader = csv.reader(f1)
        header = next(reader)
        for i in reader:
            actual_result_row.append(i)
    # print("forecast_result=", _forecast_result)
    # print("actual_result_row=", actual_result_row)

    with open(_csv_output, 'a+', encoding='gbk', newline='') as f2:
        writer = csv.writer(f2)

        headers = ['文件名', '',
                   '实际开始点击的时间点(s)', '实际加载出首页的时间点(s)', '',
                   'forecast[-3][0][0](预测B1)(s)', 'forecast[0][1][0](预测B2)(s)', 'forecast[3][0][0](预测D)(s)', '',
                   '预测B1 - 实际B(ms)', '预测B2 - 实际B(ms)', '预测D - 实际D(ms)', '',
                   '预测D - 预测B1(ms)', '实际D - 实际B1(ms)', '偏差量1(ms)', '偏差率1(%)', '',
                   '预测D - 预测B2(ms)', '实际D - 实际B2(ms)', '偏差量2(ms)', '偏差率2(%)'
                   ]

        writer.writerow(headers)
        for i in range(0, len(_forecast_result)):
            print('视频名= %s ' % _forecast_result[i][0])
            # 根据文件名，找期望结果中xxx.MP4对应的实际结果
            # 待写入csv的actual_resule
            _actual_result_temp = []
            for j in range(0, len(actual_result_row)):
                if _forecast_result[i][0] == actual_result_row[j][0]:
                    _actual_result_temp = actual_result_row[j]
                    break
                else:
                    _actual_result_temp = []
            if _actual_result_temp == []:
                print('没有找到 %s 的实际结果' % _forecast_result[i][0])
                writer.writerow([_forecast_result[i][0], 'None'])
                continue

            print('预测结果 =', _forecast_result[i])
            print('实际结果 = ', _actual_result_temp)

            # 预测B1 - 实际B
            try:
                offset1 = int(float(_forecast_result[i][1])*1000 - float(_actual_result_temp[1])*1000)
            except:
                offset1 = 'None'

            # 预测D - 预测B1
            try:
                time_interval1 = (float(_forecast_result[i][5]) - float(_forecast_result[i][1])) * 1000
            except:
                time_interval1 = 'None'

            # 实际D - 实际B1
            try:
                time_interval2 = (float(_actual_result_temp[2]) - float(_actual_result_temp[1])) * 1000
            except:
                time_interval2 = 'None'

            # (预测D - 预测B1) - (实际D - 实际B1) 预测耗时 和 实际耗时 的偏差
            try:
                offset4 = time_interval1 - time_interval2
            except:
                offset4 = 'None'

            # 预测耗时1 和 实际耗时 的偏差率
            try:
                deviation_rate1 = offset4/time_interval2*100
            except:
                deviation_rate1 = 'None'

            '''
            # 响应时间1_加载出五日横屏分时到用户开始点击的时间差
            try:
                print('\n预测加载出五日横屏分时图到用户 开始 点击的时间差 = %.1f 毫秒' % time_interval1)
            except:
                print('\n预测加载出五日横屏分时图到用户 开始 点击的时间差 = None 毫秒')
            try:
                print('实际加载出五日横屏分时图到用户 开始 点击的时间差 = %.1f 毫秒' % time_interval2)
            except:
                print('实际加载出五日横屏分时图到用户 开始 点击的时间差 None 毫秒')
            try:
                # 加载出五日分时到用户开始点击的时间差
                print('加载五日横屏分时的偏差1 = %.1f 毫秒' % offset4)
            except:
                print('加载五日横屏分时的偏差1 = None 毫秒')

            # 加载出五日分时到用户开始点击的时间差
            try:
                print('加载五日横屏分时的偏差率1 = %.1f%%\n\n' % deviation_rate1)
            except:
                print('加载五日横屏分时的偏差率1 = None\n\n')
            '''

            # 预测B2 - 实际B
            try:
                offset5 = int(float(_forecast_result[i][3])*1000 - float(_actual_result_temp[1])*1000)
            except:
                offset5 = 'None'

            # 预测D - 预测B2
            try:
                time_interval3 = (float(_forecast_result[i][5]) - float(_forecast_result[i][3])) * 1000
            except:
                time_interval3 = 'None'

            # (预测D - 预测B2) - (实际D - 实际B1) 预测耗时 和 实际耗时 的偏差
            try:
                offset6 = time_interval3 - time_interval2
            except:
                offset6 = 'None'

            # 预测耗时 和 实际耗时 的偏差率
            try:
                deviation_rate2 = offset6 / time_interval2 * 100
            except:
                deviation_rate2 = 'None'

            # '预测D - 实际D'
            try:
                offset7 = int(float(_forecast_result[i][5])*1000 - float(_actual_result_temp[2])*1000)
            except:
                offset7 = 'None'

            # 写数据到csv
            result_temp = [
                _forecast_result[i][0],
                '',
                _actual_result_temp[1], _actual_result_temp[2],
                '',
                _forecast_result[i][1], _forecast_result[i][3], _forecast_result[i][5],
                '',
                offset1, offset5, offset7,
                '',
                time_interval1, time_interval2, offset4, deviation_rate1,
                '',
                time_interval3, time_interval2, offset6, deviation_rate2,
                ]
            writer.writerow(result_temp)

    return None