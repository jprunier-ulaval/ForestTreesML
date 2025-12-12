# Identifying Forest Trees provenances using ML

## Context
Genomic tools for traceability can be powerfull to support sustainable foresttry, to fight illegal logging and improve conservation efforts. Traditional identification methods (e.g. wood anatomy, spectroscopy) are limited in resolution or scope but can be complemented with genomics approaches that can infer both species identity and geographic origins.

Machine-Learning (ML) approaches are increasingly deployed to make use of large data sets in complex models predicting an output. In our case, we developped ML models to perdict the latitude and longitude from genomics data in five forest tree species from North-America; namely _Pinus contorta_, _Pinus strobus_, _Picea mariana_, _Populus tremuloides_, _Populus trichocarpa_. We developped these models following a standard ML procedure by testing a variety of algorithms and ranges of hyper-parameters using validation and test sets. This approach was developped in Python programming language; mostly based on the scikit-learn library. The four tested algorithms that were: linear model, K-Nearest Neighbor, RandomForest and GradientBoosting.
To guide genotyping technology choices in future investigations, we additionally tested a range of SNP set sizes, from 10 SNPs to full dataset (~dozens of thousands SNPs), to evaluate the accurracy according to SNP set size.

All SNP sets, algorithms, hyper-parameters combinations were tested using a test set of 30 randomly selected samples while respecting the principle of Independent and Identically Distributed Random Variables between learning and test sets.

## Use of the script
We provide here the python script allowing to perform the analyses. It is provided 'as is' without any warranty of any kind.
This python code was first tested in a jupyter notebook and then deployed on an HPC.

The provided python script required a number of arguments


