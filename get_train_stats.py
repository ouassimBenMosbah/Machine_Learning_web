#!/usr/bin/env python
from binascii import a2b_base64
import caffe
import os
import psutil
import PIL
from PIL import Image
from subprocess import Popen, PIPE 
# -*- coding: UTF-8 -*-# enable debugging
import cgitb
import cgi
from tensorflow.python.client import device_lib
#cgitb.enable()

def get_informations():
    """result[0] contains gpu informations and result[1] cpu informations"""
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    result = [[],[]]

    if get_available_gpus():
            #If GPU available, write it datas
            p = Popen(['gpustat', '--no-color', '-c'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate("input data that is passed to subprocess' stdin")

            print("Content-Type: text/html;charset=utf-8\r\n")
            print(output.replace("\n","<br>"))

    #psutil.cpu_count(logical=True) return the number of logical CPUs available for this program
    #tot_m      Total installed memory (MemTotal and SwapTotal in /proc/meminfo)
    #used_m     Used memory (calculated as total - free - buffers - cache)
    #free_m     Unused memory (MemFree and SwapFree in /proc/meminfo)
    tot_m, used_m, free_m = os.popen('free -tmh').read().split()[-3:]

    result[1] = [psutil.cpu_count(logical=True), tot_m, used_m, free_m]

    return result
