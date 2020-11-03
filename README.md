# pmbec (Peptide to MHC binding energy covariance matrices)
### Author: Jacob Doering-Powell

generate pmbec matrices from peptide to mhc combinatorial binding assay data that can be used as scoring matrices in the training of peptide to MHC binding algorithms. The matrices can also be used to quantify how different experimental conditions contribute to overall binding from different residues.

## Requirements:
- Python 3.7+

## Installation:
Download through pypi
```pip install pmbec```

## Usage:
- example where raw data file has alleles treated with 2ME. 2ME keeps peptides that have cysteines within them in a reduced form, therefore the filtering function consolidates the impact of 2ME on cysteine residues. This shows how experimental data can be captured in a PMBEC matrix, and the general flow of how the package should be used. Order of calling should be:

1. get_raw_data()
2. filter_raw_data()
3. calculate_energy_contribution()
4. covariance()
5. cluster_matrix() - clusters the covariance matrix to see pockets of similar/dissimilar residues

```
from pmbec.pmbec_generator import pmbec_generator

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
```
### Input:
The initial input starts with loading binding data from a combinatorial binding assay between peptides and different alleles of MHC molecules. The data can be in a comma, tab, or space seperated file, but it must be specified with the sep parameter. The function also assumes that only binding data comes after the position column in the file, so if your raw data file is ordered differently, it is recommended to order the columns as such : ['column 1' , column 2', ... 'Residue column', 'Position column', all alleles binding data].

### Output:
calling covariance, and cluster_matrix, will return pandas dataframes. Any dataframe can be written with the cells colored according to the threshold set in construction as an excel file with the write_excel function.