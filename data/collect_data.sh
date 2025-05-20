#!/bin/bash

### Modified from DeepFRI

data_save_path=./
seq_clust=95

mkdir $data_save_path

printf "\n\n  downloading SIFTS-GO data...\n"
wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz -O $data_save_path/pdb_chain_go.tsv.gz

printf "\n\n  downloading SIFTS-EC data...\n"
wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_enzyme.tsv.gz -O $data_save_path/pdb_chain_enzyme.tsv.gz

printf "\n\n  downloading protein sequences from PDB...\n"
wget ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O $data_save_path/pdb_seqres.txt.gz

printf "\n\n  downloading protein clusters from PDB...\n"
wget https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-$seq_clust.txt -O $data_save_path/cbe-$seq_clust.txt

printf "\n\n  downloading GO hierarchy...\n"
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O $data_save_path/go-basic.obo

