"""
Script to extract the metric values from the training runs

Author: Tim Rosenkranz,
Goethe University, frankfurt am Main, Germany
"""

import argparse
import numpy as np

argparser = argparse.ArgumentParser(description='Metrics extraction.')
argparser.add_argument('-f', '--file', help='Input file.')
args = argparser.parse_args()
filename = args.file

inp_file = open(filename)
lines = inp_file.readlines()

ards = []
srds = []
rmses = []
rmselogs = []

params = [[], []]

for line in lines:
    if line.__contains__("ARD:"):
        value = line.split(": ")[1].replace('\n', '')
        ards.append(float(value))
    elif line.__contains__("SRD:"):
        value = line.split(": ")[1].replace('\n', '')
        srds.append(float(value))
    elif line.__contains__("RMSE:"):
        value = line.split(": ")[1].replace('\n', '')
        rmses.append(float(value))
    elif line.__contains__("RMSELog:"):
        value = line.split(": ")[1].replace('\n', '')
        rmselogs.append(float(value))
    elif line.__contains__("Total params:"):
        value = line.split(": ")[1].replace('\n', '')
        params[0].append(value)
    elif line.__contains__("Trainable params:"):
        value = line.split(": ")[1].replace('\n', '')
        params[1].append(value)

def mean(l: list):
    return sum(l) / len(l) if len(l) > 0 else None

def std(l: list, m):
    if m is None:
        return None
    sum_squares = sum([(val - m)**2 for val in l])
    return np.sqrt(sum_squares / (len(l) - 1)) / np.sqrt(len(l))

def every_second(l):
    return [l[ii] for ii in range(len(l)) if ii%2==0]

def round_to_third(val):
    return (round(val*1000)) / 1000

ard = mean(ards)
ard_std = std(ards, ard)
srd = mean(srds)
srd_std = std(srds, srd)
rmse = mean(rmses)
rmse_std = std(rmses, rmse)
rmselog = mean(rmselogs)
rmselog_std = std(rmselogs, rmselog)

print(f'Metrics for the five runs:\n ARD = {round_to_third(ard)} +/- {round_to_third(ard_std)}\n SRD = '
      f'{round_to_third(srd)} +/- {round_to_third(srd_std)}\n RMSE = {round_to_third(rmse)} +/- '
      f'{round_to_third(rmse_std)}\n RMSElog = {round_to_third(rmselog)} +/- {round_to_third(rmselog_std)}'
      f'\n --- \n Number of parameters in each run: {every_second(params[0])}\n / Trainable: {every_second(params[1])}')
