#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 00:22:03 2018

@author: abdulliaqat
"""

import os



root_path = ""

dir_list = os.listdir(root_path)

for d in dir_list:
    if(not('csv' in d)):
        temp_list = os.listdir(root_path+'/'+d)
        for f in temp_list:
            if('random_' in f):
                path = root_path+'/'+d+'/'
                os.rename(path+f,path+f.replace('random_','test_'))
