## PhiGnet for protein function prediction
Understanding protein function is crucial for medicine and biotechnology, yet many proteins remain uncharacterized. PhiGnet is a sequence-based method using statistics-informed graph networks to predict functions. The method integrates evolutionary couplings (EVCs) and residue communities (RCs) using a dual-channel architecture with stacked graph convolutional networks (GCNs). This design facilitates the assignment of functional annotations such as Enzyme Commission (EC) numbers and Gene Ontology (GO) terms (BP, CC, MF) to proteins.

### Dependencies
*PhiGnet* was developed using Python 3.6.9, and the dependencies are included in the requirement file.
Before running the PhiGnet model, one needs to install all the dependencies as follows:

`pip install -r requirement.txt`

### Download data
Download the raw data
```
bash ./data/collect_data.sh
```

### Generate evolutionary data (EVCs and RCs)
The evolutionary couplings (EVCs) and residue communities (RCs) can be obtained from the webserver 
at https://kornmann.bioch.ox.ac.uk/jang/services/amoai/index.html under "Evolutionary residue communities".

### Examples
Here are two examples to implement PhiGnet to predict protein function.

#### Example 1
```
python predict.py --ontology ec --job_id 1X37-A --dirt ./output
```

#### Example 2
```
python predict.py --ontology mf --job_id 3FM2-A --dirt ./output
```


## Get in Touch

If you have any questions, please contact QiQi Qin at [qinqiqi2021@163.com](mailto:qinqiqi2021@163.com).
