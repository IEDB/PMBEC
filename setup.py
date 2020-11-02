import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pmbec-jacobdp",
    version="0.0.1",
    author="Jacob Doering-Powell",
    author_email="jacobdoeringpowell@gmail.com",
    description="creates, visualizes, and exports peptide to MHC binding energy covariance (pmbec) matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='https://github.com/IEDB/PMBEC',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: Unix",
    ],
    package_data={
        'pmbec': [
            'reduced_cysteine_raw_data/*',
            'true_matrix/*'
        ]
    },
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'openpyxl',
        'seaborn',
        'sklearn',
        'scipy'
    ],
    python_requires='>=3.6',
)