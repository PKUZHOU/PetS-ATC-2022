import pynvml
import time

pynvml.nvmlInit()

def check_mem_util():
    # for i in range(deviceCount):
    device_id = 0 
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used
    total = meminfo.total
    util = float(used / total)
    return util

def watch(watch_interval = 0.001):
    max_util  = -1
    try:
        while(True):
            util = check_mem_util()
            max_util = max(max_util, util)
            time.sleep(watch_interval)
    except KeyboardInterrupt:
        print("Max Mem Utilization {}%:".format(max_util * 100))

watch()
