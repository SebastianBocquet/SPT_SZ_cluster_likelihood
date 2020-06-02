from __future__ import division
import numpy as np
import pickle
from cosmosis.datablock import option_section
import compute_HMF

class EmptyClass:
    pass

def setup(options):
    recalc_HMF = options.get_bool(option_section, 'recalc_HMF', default=True)
    save_HMF_to_disk = options.get_bool(option_section, 'save_HMF_to_disk', default=False)
    if recalc_HMF:
        HMF_calculator = compute_HMF.HMFCalculator(options)
    else:
        HMF_calculator = EmptyClass()
        HMF_calculator.HMF = pickle.load(open('HMF.pkl', 'rb'))
    HMF_calculator.recalc_HMF = recalc_HMF
    HMF_calculator.save_HMF_to_disk = save_HMF_to_disk
    return HMF_calculator

def execute(block, HMF_calculator):
    if HMF_calculator.recalc_HMF:
        HMF_calculator.compute_HMF(block)
        if HMF_calculator.save_HMF_to_disk:
            HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
                'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
                'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
            pickle.dump(HMF, open('HMF.pkl', 'wb'))
    else:
        block.put_double_array_1d('HMF', 'M_arr', HMF_calculator.HMF['M_arr'])
        block.put_double_array_1d('HMF', 'z_arr', HMF_calculator.HMF['z_arr'])
        block.put_double_array_nd('HMF', 'dNdlnM', HMF_calculator.HMF['dNdlnM'])
    return 0

def cleanup(config):
    pass
