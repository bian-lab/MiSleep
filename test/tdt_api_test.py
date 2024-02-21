# -*- coding: UTF-8 -*-
"""
@Project: MiSleep 
@File: tdt_api_test.py
@IDE: PyCharm 
@Author: Xueqiang Wang
@Date: 2024/1/19 21:01 
@Description:  
"""
import tdt

data = tdt.read_block(r'E:\workplace\EEGProcessing\00_DATA\WXQ_20240112_TEST_tdt')
print(type(data))
