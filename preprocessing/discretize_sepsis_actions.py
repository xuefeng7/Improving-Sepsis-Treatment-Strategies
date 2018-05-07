import numpy as np
import pandas as pd

def discretize_actions(
        input_4hourly__sequence__continuous,
        median_dose_vaso__sequence__continuous,
        bins_num = 5):
    
    # IV fluids discretization
    input_4hourly__sequence__continuous__no_zeros = input_4hourly__sequence__continuous[ \
        input_4hourly__sequence__continuous != 0]
    input_4hourly__sequence__discretized__no_zeros, input_4hourly__bin_bounds = \
        pd.qcut( input_4hourly__sequence__continuous__no_zeros, \
                 bins_num - 1, labels = False, retbins = True)
    input_4hourly__sequence__discretized = \
        (input_4hourly__sequence__continuous != 0).astype(int)
    input_4hourly__sequence__discretized[ input_4hourly__sequence__discretized == 1 ] = \
        input_4hourly__sequence__discretized__no_zeros + 1
        
    # Vaopressors discretization
    median_dose_vaso__sequence__continuous__no_zeros = median_dose_vaso__sequence__continuous[ \
        median_dose_vaso__sequence__continuous != 0]
    median_dose_vaso__sequence__discretized__no_zeros, median_dose_vaso__bin_bounds = \
        pd.qcut( median_dose_vaso__sequence__continuous__no_zeros, \
                 bins_num - 1, labels = False, retbins = True)
    median_dose_vaso__sequence__discretized = \
        (median_dose_vaso__sequence__continuous != 0).astype(int)
    median_dose_vaso__sequence__discretized[ median_dose_vaso__sequence__discretized == 1 ] = \
        median_dose_vaso__sequence__discretized__no_zeros + 1
        
    # Combine both actions discretizations
    actions_sequence = median_dose_vaso__sequence__discretized * bins_num + \
        input_4hourly__sequence__discretized
    
    # Calculate for IV fluids quartiles the median dose given in that quartile
    input_4hourly__conversion_from_binned_to_continuous = np.zeros(bins_num)
    for bin_ind in range(1, bins_num):
        input_4hourly__conversion_from_binned_to_continuous[bin_ind] = \
        np.median(input_4hourly__sequence__continuous__no_zeros[ \
                  input_4hourly__sequence__discretized__no_zeros == bin_ind - 1] )
    
    # Calculate for vasopressors quartiles the median dose given in that quartile
    median_dose_vaso__conversion_from_binned_to_continuous = np.zeros(bins_num)
    for bin_ind in range(1, bins_num):
        median_dose_vaso__conversion_from_binned_to_continuous[bin_ind] = \
        np.median(median_dose_vaso__sequence__continuous__no_zeros[ \
                  median_dose_vaso__sequence__discretized__no_zeros == bin_ind - 1] )
    
    return actions_sequence, \
        median_dose_vaso__conversion_from_binned_to_continuous,\
        input_4hourly__conversion_from_binned_to_continuous
        