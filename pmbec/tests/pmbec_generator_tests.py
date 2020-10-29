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
    #assert that there are other positions besides 2 and 9 before filtering
    positions_set = set([])
    for aa in raw_data:
        for pos in raw_data[aa]:
            positions_set.add(pos)
    assert len(positions_set) > 2
    assert 1 in positions_set
    #assert that 2ME is in alleles before filtering
    twoME_present = False
    for aa in raw_data:
        for pos in raw_data[aa]:
            for allele in raw_data[aa][pos]:
                if '2ME' in allele:
                    twoME_present = True
                    break
    assert twoME_present
    raw_data = pm.filter_raw_data(raw_data, 
                                  consolidate=True, 
                                  consolidate_on='2ME',
                                  positions={2,9}, 
                                  residues_to_consolidate=set(['C']),
                                  skip_alleles='2ME')
    #assert that 2ME is not in alleles after filtering
    twoME_present = False
    for aa in raw_data:
        for pos in raw_data[aa]:
            for allele in raw_data[aa][pos]:
                if '2ME' in allele:
                    twoME_present = True
    assert not twoME_present
    #assert only posoitions 2 and 9 are in raw data after filtering
    positions_set = set([])
    for aa in raw_data:
        for pos in raw_data[aa]:
            positions_set.add(pos)
    assert len(positions_set) == 2
    for i in range(10):
        if i != 2 and i != 9:
            assert i not in positions_set
        else:
            assert i in positions_set
    assert pm.filtered == True
    #assert getting a new raw_data set ressets the filtered flag
    raw_data = pm.get_raw_data("../reduced_cysteine_raw_data/Cysteine_surrogate_raw_data.csv",
                    'Residue',
                    'Position',
                    nrows=49,
                    sep=',')
    assert pm.filtered == False
    tear_down(pm.job_id)

def test_energy_contribution():
    pm = setup_pm()
    exception_thrown = False
    #assert that exception is thrown properly
    try:
        pm.calculate_energy_contribution()
    except Exception as e:
        exception_thrown = True
    assert exception_thrown
    raw_data = pm.get_raw_data("../reduced_cysteine_raw_data/Cysteine_surrogate_raw_data.csv",
                    'Residue',
                    'Position',
                    nrows=49,
                    sep=',')
    # check that if filter is not called before callig calculating energy contribution
    # then the unfiltered and filtered raw_data csv are the same
    energy_contribution = pm.calculate_energy_contribution()
    unfiltered_file = os.getcwd() + '/' + pm.job_id + '/' + pm.job_id + '_unfiltered_raw_data.csv'
    filtered_file = os.getcwd() + '/' + pm.job_id + '/' + pm.job_id + '_filtered_raw_data.csv'
    filtered_data = pm.read_intermediate_file(filtered_file)
    unfiltered_data = pm.read_intermediate_file(unfiltered_file)
    for filtered_aa, unfiltered_aa in zip(sorted(filtered_data.keys()), sorted(unfiltered_data.keys())):
        assert filtered_aa == unfiltered_aa
        for filtered_position, unfiltered_position in zip(sorted(filtered_data[filtered_aa].keys()), sorted(unfiltered_data[unfiltered_aa].keys())):
            assert filtered_position == unfiltered_position
            filtered_alleles = sorted(list(filtered_data[filtered_aa][filtered_position].keys()))
            unfiltered_alleles = sorted(list(unfiltered_data[unfiltered_aa][unfiltered_position].keys()))
            for filtered_allele, unfiltered_allele in zip(filtered_alleles, unfiltered_alleles):
                assert filtered_allele == unfiltered_allele
    filtered_data = pm.filter_raw_data(raw_data, 
                                  consolidate=True, 
                                  consolidate_on='2ME',
                                  positions={2,9}, 
                                  residues_to_consolidate=set(['C']),
                                  skip_alleles='2ME')
    energy_contribution = pm.calculate_energy_contribution()
    filtered_data = pm.read_intermediate_file(filtered_file)
    assert len(filtered_data.keys()) != len(unfiltered_data.keys())
    tear_down(pm.job_id)
    
                

