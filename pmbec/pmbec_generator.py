from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import re
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial as sp
from matplotlib import pyplot as plt
from pathlib import Path
import os

class pmbec_generator():
    '''
    Class has the ability to create Peptide to MHC Binding Energy Covariance (PMBEC)
    matrices, perform analysis with them, export them, and create plots with them. 
    PMBEC matrices are useful because they can be used as scoring matrices in peptide 
    to MHC binding algorithms. 

    :Example usage:
    import pmbec_generator

    pm = pmbec_generator(job_id='pmbec_matrix', threshold=.05)
    raw_data = pm.get_raw_data("raw_data_file",
                'Residue',
                'Position',
                nrows=49,
                sep=',')
    pm.filter_raw_data(raw_data,
                       consolidate=True,
                       consolidate_on='2ME',
                       positions={2,9},
                       residues_to_consolidate=set(['C']),
                       skip_alleles='2ME')
    energy_contribution = pm.calculate_energy_contribution()
    cov_matrix = pm.covariance()
    clustered = pm.cluster_matrix(cov_matrix)
    pm.write_excel(clustered)

    '''
    def __init__(self, threshold=0.05, job_id='pmbec'):
        '''
        constructor for pmbec matrix generator

        :param: threshold - float - this threshold will be used to color PMBEC matrices later
                job_id - str - default 'pmbec' - This is a job id label for creating directories where
                                                    intermediate files will be stored, along with the final matrix
        '''
        if not isinstance(job_id, str):
            raise Exception("job_id must be a string, it will be used as a directory to write intermediate files and covariance matrices to")
        if not isinstance(threshold, float):
            raise Exception("threshold must be a float, and will be used as a metric to color the final matrix")
        self.job_id = job_id
        self.threshold = threshold
        self.filtered = False
        self.pmbec_matrix = None
        self.energy_contribution_file = None
        self.raw_data_file = None

    def load_true_matrix(self, true_matrix_file='./true_matrix/covariance_matrix.mat', sep=' '):
        index = "A C D E F G H I K L M N P Q R S T V W Y"
        index = index.split(" ")
        initial_matrix = pd.read_csv(true_matrix_file, sep=sep,header=0,index_col=0)
        new_columns = {"Unnamed: 0": "A",
                "A": "C",
                "C": "D",
                "D": "E",
                "E": "F",
                "F": "G",
                "G": "H",
                "H": "I",
                "I": "K",
                "K": "L",
                "L": "M",
                "M": "N",
                "N": "P",
                "P": "Q",
                "Q": "R",
                "R": "S",
                "S": "T",
                "T": "V",
                "V": "W",
                "W": "Y",
                "Y": "Unnamed: 0"
                }
        true_matrix = initial_matrix.rename(columns=new_columns)
        true_matrix = true_matrix.drop(columns=["Unnamed: 0"])
        clustered = self.cluster_matrix(true_matrix, title='Dendrogram from original SMM-PMBEC covariance matrix')
        return clustered

    def write_intermediate_file(self, data_structure, file_name):
        residues = set(data_structure.keys())
        #first map out residues:(allele, position)
        #then write out first and second row 
        #then for each residue, get all its allele, positions, add (string) ic50 to a list
        #write out list to file
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(file_name, 'w+')
        allele_position_set = set()
        residue_to_alleles = defaultdict(set)
        for r in residues:
            for pos in data_structure[r].keys():
                for allele in data_structure[r][pos].keys():
                    tuple_ = (allele, str(pos))
                    allele_position_set.add(tuple_)
                    residue_to_alleles[r].add(tuple_)
        allele_and_positions = list(allele_position_set)
        alleles = []
        positions = []
        for a_p in allele_and_positions:
            alleles.append(a_p[0])
            positions.append(a_p[1])
        alleles = [''] + alleles + ['\n']
        positions = [''] + positions + ['\n']
        f.write(','.join(alleles))
        f.write(','.join(positions))
        for r in residues:
            row = [r]
            for a_p in allele_and_positions:
                if a_p in residue_to_alleles[r]:
                    row.append(str(data_structure[r][int(a_p[1])][a_p[0]]))
                else:
                    row.append('') #case where there is no value at the position/allele
            row.append('\n')
            f.write(','.join(row))
        f.close()

    def read_intermediate_file(self, file_name, return_dataframe=False):
        matrix = pd.read_csv(file_name)
        matrix = matrix.drop(matrix.columns[-1], axis=1)
        if return_dataframe:
            matrix = matrix.rename(columns={"Unnamed: 0":'residues'})
            values = matrix.iloc[0]
            new_columns = {}
            for c,v in zip(matrix.columns, values):
                if np.isnan(v):
                    new_columns[c] = c
                else:
                    new_columns[c] = '(' + c + ', ' + str(int(v)) + ')'
            matrix = matrix.iloc[1:]
            matrix = matrix.rename(columns=new_columns)
            matrix = matrix.set_index('residues')
            return matrix
        mhcs = matrix.columns[1:]
        matrix = matrix.transpose()
        new_columns = list(matrix.iloc[0])
        new_columns[0] = 'position'
        column_map = {i:c for i,c in enumerate(new_columns)}
        matrix = matrix.drop(matrix.index[0])
        matrix = matrix.rename(index=str, columns=column_map)
        data_structure = {}
        for row in matrix.itertuples():
            allele = ''
            position = None
            for i,v in enumerate(row):
                if i == 0:
                    allele = row[i]
                elif i == 1:
                    position = int(row[i])
                else:
                    residue = new_columns[i - 1]
                    if residue not in data_structure:
                        data_structure[residue] = defaultdict(dict)
                    if position not in data_structure[residue]:
                        data_structure[residue][position] = defaultdict(dict)
                    data_structure[residue][position][allele] = row[i]
        return data_structure
        
    def consolidate_dataset(self, all_raw_data, residues=set(['C']), consolidate_on='2ME',pos=set([2,9])):
        '''
        consolidation function that takes raw data dictionary of <residues<positions<MHC:IC50>>> and
        consolidates a 'new' in-silico residue based on whatever parameter is passed in with 'consolidate_on'.
        This means that if a user has experimental conditions marked in their binding assays that they want to capture in the covariance
        matrix, they can do so. 

        :param: all_raw_data - Dict(<str<int<str:float>>>) - raw data containing <Residues<positions<MHC alleles : IC50 values>>>
                residues - Set<str> - a set of residues that will be used to create the new in silico residue
                consolidate_on - str - a string that marks what experimental condition in the binding assay is going to be captured
                pos - List<int> - a list of positions to check for experimental conditions
        :return: raw_data - Dict(<str<int<str:float>>>) - a raw data dictionary with the new consolidated residue
        '''
        raw_data = all_raw_data
        for r in residues:
            r_prime = r + " \'"
            #all_raw_data['residues'].add(r_prime)
            self.residues.add(r_prime)
            raw_data[r_prime] = defaultdict(int)
            positions = raw_data[r]
            for p in positions.keys():
                if pos:
                    if p not in pos:
                        continue
                raw_data[r_prime][p] = defaultdict(dict)
                for allele in raw_data[r][p].keys():
                    if allele.find(consolidate_on) != -1:
                        index = allele.find(consolidate_on)
                        allele_without_string = allele[:index]
                        allele_without_string = allele_without_string.rstrip()
                        if allele_without_string in raw_data[r][p]: #only add in substitutes that have the allele, can be changed
                            raw_data[r_prime][p][allele_without_string] = raw_data[r][p][allele]
        return raw_data

    def remove_skip_residues(self, all_raw_data, skip_residues):
        for c in skip_residues:
            all_raw_data.pop(c, None)
            self.residues.discard(c)
        return all_raw_data

    def filter_raw_data(self,
                    raw_data_dict,
                    consolidate=False,
                    residues_to_consolidate=set(),
                    consolidate_on=None,
                    skip_residues=set(),
                    skip_alleles=None,
                    positions=set(),
                    *args,
                    **kwargs
                    ):
        '''
        Function filters raw data before analysis begins. It allows a user to consolidate parameters
        found in the raw data in an in-silico amino acid so the parameters effect can be seen in the PMBEC
        matrix. For instance, all alleles could be treated with 2ME in the raw data, and 2ME prevents cysteine
        from oxidizing. To capture the effect of 2ME in the PMBEC matrix, one could set consolidate to true, and
        pass in the residues that are impacted by 2ME, which in this case would be cysteine.

        :param: raw_data_dict - dict(<str<int<str:float>>>) explicitly <residue<position<allele:IC50>>>
                consolidate - default False - flag to signal consolidation be called
                residues_to_consolidate - default empty set - set of residues that parameters in raw data are being consolidated on
                consolidate_on - default None - a str of the parameter that is being consolidated on in the raw dataset
                skip_residues - default empty set - a set of residue strings to skip
                skip_alleles - default None - a string that describes the allele to be skipped
                positions - default empty set - a set of positions that describe the positions to be kept.
                                                if empty set, all positions found will be kept.
        :return: filtered raw_data_dict <str<int<str:float>>> explicitly <residue<position<allele:IC50>>>
        '''
        if consolidate: # check to consolidate any parameters in raw data
            if not consolidate_on:
                raise Exception("you must pass in a parameter found in the raw data to consolidate on")
            if len(residues_to_consolidate) == 0:
                print('WARNING: you have not passed in any residues to consolidate on, pass in a set of residues that the parameter impacts')
            raw_data_dict = self.consolidate_dataset(raw_data_dict, residues=residues_to_consolidate, consolidate_on=consolidate_on)
        if len(skip_residues) > 0: # check to see if any residues are being filtered out
            raw_data_dict = self.remove_skip_residues(raw_data_dict, skip_residues)
        if len(positions) > 0: #filter out any positions
            self.number_positions = len(positions)
            new_rd = {}
            for r in raw_data_dict.keys():
                new_rd[r] = defaultdict(int)
                for pos in raw_data_dict[r].keys():
                    if pos in positions:
                        new_rd[r][pos] = {}
                        for allele in raw_data_dict[r][pos].keys():
                            new_rd[r][pos][allele] = raw_data_dict[r][pos][allele]
            raw_data_dict = new_rd
        else:
            self.number_positions = 9
        if skip_alleles: # filter out any alleles
            if not isinstance(skip_alleles, str):
                raise Exception('Improper type, skip_alleles should be a string that describes the alleles to be skipped')
            new_rd = {}
            new_rd = defaultdict()
            for r in raw_data_dict.keys():
                new_rd[r] = defaultdict(int)
                for pos in raw_data_dict[r].keys():
                    new_rd[r][pos] = {}
                    for allele in raw_data_dict[r][pos].keys():
                        if allele.find(skip_alleles) == -1:
                            new_rd[r][pos][allele] = raw_data_dict[r][pos][allele]
                        else:
                            if allele in self.mhcs:
                                self.mhcs.remove(allele)
            raw_data_dict = new_rd
        self.raw_data = raw_data_dict
        self.raw_data_file = os.getcwd() + '/' + self.job_id + '/' + self.job_id + '_filtered_raw_data.csv'
        self.write_intermediate_file(self.raw_data, self.raw_data_file)
        self.filtered = True
        return raw_data_dict

    def get_raw_data(self, 
                    raw_data_file, 
                    residue_column_string, 
                    position_column_string,
                    nrows=180,
                    sep="\t",
                    write_unfiltered=True,
                    *args,
                    **kwargs):
        '''
        This method simply takes in a raw data file and loads it into a series of dictionaries to perform downstream
        analysis on. This function assumes that the raw data file has columns structured in this order [peptide, sequence, residues, positions, all MHC molecules].
        Essentially it assumes that after the position column in the raw data there is only binding information for MHC molecules. 

        :param: residue_column_string - str - the name of the column in the raw data file that contains all the residues present in binding experiments
                position_column_string - str - the name of the column that contains positions of the residues
                nrows - int - default 180 -  number of rows in the dataset that contain useful information
                sep - str - default is tab seperated - how the raw data file is delimited, examples: tabs, spaces, commas
                wrtie_unfiltered - bool - default True - writes unfiltered data to the job id directory for debugging purposes if necessary
        :return: unfiltered nested raw data dictionary of types <str<int<str:float>>> in explicit terms <residue<positions<MHC molecules : IC50>>>
        '''
        raw_data = pd.read_csv(raw_data_file, sep=sep, nrows=nrows)
        residues = set(raw_data[residue_column_string].tolist())
        position_index = raw_data.columns.get_loc(position_column_string) #assumption is alleles come after position column
        mhcs = list(raw_data.columns)[position_index+1:]
        raw_data_dict = defaultdict(dict)
        for i,row in raw_data.iterrows():
            r = row[residue_column_string]
            pos = row[position_column_string]
            for mhc in mhcs:
                    ic50 = row[mhc]
                    if isinstance(ic50, str):
                        ic50 = float(ic50.strip().replace('<','').replace(',',''))
                    if not ic50:
                        print("NaN found at " + str(r) + " position " + str(pos) + ' allele ' + str(mhc)) 
                    if not pd.isna(ic50): #known value, but in dictionary
                        if r not in raw_data_dict:
                            raw_data_dict[r] = defaultdict(dict)
                        if pos not in raw_data_dict[r]:
                            raw_data_dict[r][pos] = defaultdict(dict)
                        raw_data_dict[r][pos][mhc] = ic50
        self.mhcs = mhcs
        self.residues = residues
        self.raw_data = raw_data_dict
        self.filtered = False
        self.raw_data_file = os.getcwd() + '/' + self.job_id + '/' + self.job_id + '_unfiltered_raw_data.csv'
        if write_unfiltered:
            self.write_intermediate_file(self.raw_data, self.raw_data_file)
        return raw_data_dict

    def get_ic50(self, aa, pos, MHC, raw_data_dict):
        return raw_data_dict[aa][pos][MHC]

    def normalized_energy_contribution_sum(self, aa, pos, MHC, raw_data_dict):
        '''
        This function gets the normalized energy contribution for a given position. This is necessary for the 
        energy contribution calculation itself.

        :param: pos - an integer representing the position the residue in the peptide kmer :type: int
                MHC - the allele of the MHC given in order get a value from the raw_data_dictionary :type: str
                raw_data_dict - a dictionary, of dictionaries, of dictionaries. the first key is the residue, the next key
                                is a position, and the last key is a MHC allele mapping to an ic50 binding value :type: <dict<dict<str:int>>>
        :return: the normalized energy contribution sum for every residue at a given position
        '''
        sum_ = 0
        for r in raw_data_dict.keys():
            if pos in raw_data_dict[r]:
                ic50 = self.get_ic50(r, pos, MHC, raw_data_dict)
                if isinstance(ic50, float) or isinstance(ic50, int):
                    sum_ += np.log10(ic50)
                else:
                    raise Exception("type mismatch, IC50 values need to be floats or ints, not " + str(type(ic50)))
        return (1/len(raw_data_dict.keys()))*sum_

    def energy_cont(self, aa, pos, MHC, raw_data_dict):
        '''
        Method calculates the energy contibution for each amino acid, position, and MHC allele

        :param: aa - an amino acid residue string :type: str
                pos - an integer representing the position the residue in the peptide kmer :type: int
                MHC - the allele of the MHC given in order get a value from the raw_data_dictionary :type: str
                raw_data_dict - a dictionary, of dictionaries, of dictionaries. the first key is the residue, the next key
                                is a position, and the last key is a MHC allele mapping to an ic50 binding value :type: <dict<dict<str:int>>>
        :return: the energy contribution of a given amino acid, at a given position, at a given allele
        '''
        ic50 = self.get_ic50(aa, pos, MHC, raw_data_dict)
        if isinstance(ic50, float) or isinstance(ic50, int):
            log_ic50 = np.log10(ic50)
        else:
            raise Exception("improper type")
        return log_ic50 - self.normalized_energy_contribution_sum(aa, pos, MHC, raw_data_dict)
    
    def calculate_energy_contribution(self, raw_data=None):
        '''
        function calculates the relative energy contribution for each residue found in the raw data.
        it then writes this analysis to an intermediate file. After computing the energy contribution
        for each residue, position, and allele, a user can calculate the covariance. This function assumes
        that get_raw_data() has been called. If a user wants special filtering done to the raw data they should call
        filter_data() with the parameters wanted. If it is not called the energy contribution will be calculated with
        the raw data as is. There is a parameter for raw_data to be calculated if a user does not want to read in the raw_data.
        
        :param: raw_data - dict(<str<int<str : float>>>) -  default None - if a user does not want the filtered raw data to be calculated on
                                                                          they can pass in a dictionary of <residue<position<MHC alleles : IC50>>>
        
        :return: energy contribution dictonary of <str<int<str : float>>> or in other words <residue<position<MHC alleles : relative binding affinity>>>
        '''
        # if not self.raw_data_file:
        #     raise Exception("Must filter raw data before calculating energy contribution")
        if not raw_data:
            if not self.raw_data_file:
                raise Exception("must first read in and filter raw data before calculating energy contribution")
            raw_data_dict = self.read_intermediate_file(self.raw_data_file)
            if not self.filtered:
                print('WARNING: calculating energy contribution on unfiltered raw data, filtered_raw_data.csv will be the same as unfiltered_raw_data.csv')
                self.filter_raw_data(raw_data_dict)
        else:
            raw_data_dict = raw_data
        energy_contribution_dict = defaultdict(dict)
        for r in raw_data_dict.keys():
            energy_contribution_dict[r] = defaultdict(dict)
            for pos in raw_data_dict[r].keys():
                energy_contribution_dict[r][pos] = defaultdict(dict)
                for mhc in raw_data_dict[r][pos].keys():
                    energy_contribution_dict[r][pos][mhc] = self.energy_cont(r, pos, mhc, raw_data_dict)
        self.energy_contribution_file = os.getcwd() + '/' + self.job_id + '/' + self.job_id + '_energy_contribution.csv'
        self.write_intermediate_file(energy_contribution_dict, self.energy_contribution_file)
        return energy_contribution_dict

    def cov(self, x, y, population_covariance=True):
        if population_covariance:
            covariance = np.cov(x, y, bias=True)[0][1]
        else:
            covariance = np.cov(x, y)[0][1]
        return covariance
    
    def covariance(self, population_covariance=True, raw_data=None):
        if not self.energy_contribution_file:
            print('covariance() called before calling calculate_energy_contribution()\ncalculating energy contribution now')
            self.calculate_energy_contribution(raw_data=raw_data)
        energy_contribution_df = self.read_intermediate_file(self.energy_contribution_file, return_dataframe=True)
        covariance_dict = {}
        for residue in self.residues:
            covariance_dict[residue] = {}
            for r in self.residues:
                x = energy_contribution_df.loc[residue]
                y = energy_contribution_df.loc[r]
                covariance_dict[residue][r] = self.cov(x,y, population_covariance=population_covariance)
        matrix = pd.DataFrame.from_dict(covariance_dict)
        self.pmbec_matrix = matrix
        return matrix

    def cluster_matrix(self, df, cluster_map=True, plot=True, title=None):
        Z = linkage(df.values, 'complete', optimal_ordering=True, metric='euclidean')
        dn = dendrogram(Z, labels=df.columns)
        optimal_ordering = dn['ivl']
        new_columns = df[optimal_ordering]
        new_df = new_columns.reindex(optimal_ordering)
        self.pmbec_matrix = new_df
        if cluster_map:
            heatmap = sns.clustermap(new_df, method='complete')
        plt.clf()
        plt.cla()
        plt.close()
        return new_df

    def color_matrix(self, matrix):
        matrix = matrix.round(decimals=3)
        return matrix.style.applymap(self._color_cells)

    def _color_cells(self, val):
        if val <= -1 * self.threshold:
            color = 'red'
        elif val >= self.threshold:
            color = 'green'
        else:
            color = 'white'
        return 'background-color: %s' % color

    def write_excel(self, matrix):
        file_name = os.getcwd() + '/' + self.job_id + '/' + self.job_id + '_pmbec_matrix.xlsx'
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_name:
            if file_name.split('.')[-1] == 'xlsx':
                self.color_matrix(matrix).to_excel(file_name)
            else:
                self.color_matrix(matrix).to_excel(file_name + '.xlsx')
        else:
            raise Exception("pass in a file name or matrix to write")

    def query_pmbec_matrix(self, row_amino_acid, column_amino_acid):
        '''
        function indexes the PMBEC matrix by two amino acids to get the pairs' binding energy covariance value.

        :param: row_amino_acid - str - amino acid to index the row of the PMBEC matrix
                column_amino_acid - str - amino acid to index the column of the PMBEC matrix
    
        :return: a float representing the covariance value of the pair
        '''
        if not self.pmbec_matrix:
            raise Exception('call covariance() to create the PMBEC matrix before you query a matrix')
        if not isinstance(row_amino_acid, str) or not isinstance(column_amino_acid, str):
            raise Exception("row_amino_acid and column_amino_acid need to be strings")
        if row_amino_acid in list(self.pmbec_matrix.index) and column_amino_acid in list(self.pmbec_matrix.columns):
            return self.pmbec_matrix.loc[row_amino_acid, column_amino_acid]
        else:
            if row_amino_acid not in list(self.pmbec_matrix.index):
                raise Exception(row_amino_acid + ' was not found in any row')
            if column_amino_acid not in list(self.pmbec_matrix.columns):
                raise Exception(column_amino_acid + ' was not found in any column')
    
    def rename_residue(self, old_residue, new_residue):
        '''
        function renames a column and row (index) of the pmbec matrix.

        :param: old_residue - str - the old row and column name of the residue
                new_residue - str - the new row and column name of the residue
        :return: covariance matrix dataframe with a renamed residue
        '''
        if not self.pmbec_matrix:
            raise Exception('create a PMBEC matrix before trying to rename residues')
        self.pmbec_matrix = self.pmbec_matrix.rename(columns={old_residue: new_residue}, index={old_residue: new_residue})
        return self.pmbec_matrix