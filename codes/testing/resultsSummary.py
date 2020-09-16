import os

import pandas as pd
import sys
import re
import numpy as np

kSTART = 0
kEVAL_START = 1
kEVAL_END = 1


def extractStats(l_idx, log_lines):
    result = re.search(r"([\d.]*) dB.* ([\d.]*)", log_lines[l_idx])
    psnr = float(result.group(1))
    ssim = float(result.group(2))

    l_idx += 3
    result = re.search(r"([\d.]*) dB.* ([\d.]*)", log_lines[l_idx])
    psnr_y = float(result.group(1))
    ssim_y = float(result.group(2))

    return [psnr, ssim, psnr_y, ssim_y]


def advanceCounter(steper_c):

    for i in range(len(steper_c) - 1, -1, -1):
        steper_c[i] += 0.1
        if steper_c[i] > 1:
            steper_c[i] = 0
        else:
            break
    return steper_c


def steper2str(steper_c, steper_nam):
    str = ""
    for s_name, s_count in zip(steper_nam, steper_c):
        str += "{}_{:.1f}_".format(s_name, s_count)
    return str[:-1]


def main():
    file_path = sys.argv[1]
    stepers_names = sys.argv[2].split(',')
    stepers_n = len(stepers_names)
    steper_c = np.zeros(stepers_n)

    with open(file_path, 'r') as log_pd:
        log_lines = log_pd.readlines()

    summary = dict()
    state = kSTART
    idx = 0
    while idx < len(log_lines):
        line = log_lines[idx]
        idx += 1

        if state is kSTART:
            if 'INFO: 0000' in line:
                state = kEVAL_START
            else:
                continue
        if state is kEVAL_START:
            if "Average PSNR/SSIM results for" in line:
                state = kEVAL_END
            else:
                continue
        if state is kEVAL_END:
            state = kEVAL_START

            score_str = extractStats(idx, log_lines)
            idx += 4

            summary[steper2str(steper_c, stepers_names)] = score_str
            steper_c = advanceCounter(steper_c)

    with open(os.path.dirname(file_path) + '/summary.csv', 'w') as summ_pd:
        summ_pd.write("KEY,PSNR,SSIM,PSNR_Y,SSIM_Y\n")
        for k, v in summary.items():
            summ_pd.write(','.join([k, *[str(x) for x in v]]))
            summ_pd.write('\n')



if __name__ == '__main__':
    main()
