from pmbec.pmbec_generator import *
import shutil
import sys
import pytest
import os
sys.path.append("..")

def setup_pm():
    pm = pmbec_generator(job_id='in_tests_directory')
    return pm

def tear_down(path):
    shutil.rmtree(path)

def test_pm_creation():
    pm = setup_pm()
    assert pm.filtered == False
    assert pm.threshold == 0.05
    assert pm.pmbec_matrix == None
    assert pm.raw_data_file == None
    assert pm.energy_constribution_file == None

def test_pm_get_raw_data():
    pm = setup_pm()
    assert os.path.isdir(pm.job_id) == False
    raw_data = pm.get_raw_data("../reduced_cysteine_raw_data/Cysteine_surrogate_raw_data.csv",
                    'Residue',
                    'Position',
                    nrows=49,
                    sep=',')
    assert os.path.isdir(pm.job_id) == True
    assert pm.raw_data_file == os.getcwd() + '/' + pm.job_id + '/' + pm.job_id + '_unfiltered_raw_data.csv'
    assert pm.raw_data != None
    tear_down(pm.job_id)

def test_pm_filter_data():
    pm = setup_pm()
    raw_data = pm.get_raw_data("../reduced_cysteine_raw_data/Cysteine_surrogate_raw_data.csv",
                    'Residue',
                    'Position',
                    nrows=49,
                    sep=',')
    pm.filter_raw_data(raw_data, consolidate=True, positions={2,9}, skip_alleles='2ME')
    # TODO ADD TO THIS TOMORROW
    assert pm.filtered == True
    raw_data = pm.get_raw_data("../reduced_cysteine_raw_data/Cysteine_surrogate_raw_data.csv",
                    'Residue',
                    'Position',
                    nrows=49,
                    sep=',')
    assert pm.filtered == False
    tear_down(pm.job_id)