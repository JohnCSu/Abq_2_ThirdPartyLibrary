
from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import *
import os
import numpy as np
from functions import write_csv,read_csv,format_path_number,calculate_plasticity
import math

openMdb('pillar.cae')
params_file = 'params.csv'

wd = os.getcwd()

if not os.path.exists('params.csv'):
    params = {
        'K':0.3E9,
        'n': 3.00
    }
    write_csv(params_file,params)

params = read_csv(params_file)

plasticity = calculate_plasticity(**params)

mdb.models['column'].materials['STEEL'].plastic.setValues(scaleStress=None, table=(plasticity))

#Submit Job
#Set up directories
if not os.path.exists('Results'):
    os.mkdir('Results')

file_name = f'n_{format_path_number(params["n"],"1.3f")}_K_{format_path_number(params["K"],"1.3E")}'
# file_name = f"radius_{ format_path_number(params['radius'],format_code='1.3E' ) }_elastic_{format_path_number(params['elasticity'],'1.3E')}}"

n_runs = len(os.listdir('Results'))
folder_path=f'Results/{n_runs}'
if not os.path.exists(folder_path):
    os.mkdir(path =folder_path)
    
# Job Submission
os.chdir(folder_path)
job = mdb.Job(name=file_name, model='column', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
    contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
    resultsFormat=ODB, numDomains=1, activateLoadBalancing=False, 
    numThreadsPerMpiProcess=1, multiprocessingMode=DEFAULT, numCpus=1)

job.submit(consistencyChecking=OFF)
job.waitForCompletion()

o1 = openOdb(path=f'{file_name}.odb')
step = o1.steps['Step-1']
region_keys = step.historyRegions.keys()
force = [step.historyRegions[key].historyOutputs['RF2'].data for key in region_keys if  'RIGIDPILLARS' in str(key)][0]
write_csv('params.csv',params)
x = np.array(force)
with open('force.csv','w') as f:
    np.savetxt(f,x,delimiter= ',')
os.chdir(wd)


