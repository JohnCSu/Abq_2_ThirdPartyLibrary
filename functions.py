import csv
import numpy as np
def write_csv(filename,params_dict):
    with open(filename,'w',newline='') as f:
        csv_writer = csv.writer(f)
        # print(list(params_dict.values()))
        values = [float(val) for val in params_dict.values()]
        csv_writer.writerows( [list(params_dict.keys()), values])


def read_csv(filename):
    with open(filename,'r',newline='') as f:
        csv_reader = csv.reader(f)
        headers,params = [x for x in csv_reader]
        return {name:float(value) for name,value in zip(headers,params)}
    

def format_path_number(n,format_code = '1.3f'):
    n = f'{n:{format_code}}'
    n = n.replace('.','_')
    n = n.replace('+','')
    return n

def calculate_plasticity(K,n):
    stresses = np.loadtxt('hardening.csv')
    plastic_strain = (stresses/K)**n
    plastic_strain[0] = 0.0
    return tuple( [(stress,pE) for stress,pE in zip(stresses,plastic_strain)])



def decimate(x,keep_first = 20,decimate = 10):
    x_first = x[:keep_first]
    x_next = x[keep_first:][::decimate]

    return np.concatenate([x_first,x_next])


def interp1D_func(xp,yp):
    def inner(x):
        return np.interp(x,xp,yp)
    return inner