DEFT
=====

This repository contains the code for [Learnable Spectral Wavelets on Dynamic Graphs to Capture Global Interactions](https://arxiv.org/abs/2211.11979), published in AAAI 2023.

## Data

8 datasets were used in the paper:

- stochastic block model: See the 'data' folder. Untar the file for use.
- bitcoin OTC: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- uc_irvine: Downloadable from http://konect.uni-koblenz.de/networks/opsahl-ucsocial
- autonomous systems: Downloadable from http://snap.stanford.edu/data/as-733.html
- reddit hyperlink network: Downloadable from http://snap.stanford.edu/data/soc-RedditHyperlinks.html
- elliptic: Please see the [instruction](elliptic_construction.md) to manually prepare the preprocessed version or refer to the following repository that originally proposed the usage of the data: https://arxiv.org/abs/1902.10191
- brain: Downloadable from https://www.dropbox.com/sh/33p0gk4etgdjfvz/AACe2INXtp3N0u9xRdszq4vua?dl=0
 
For downloaded data sets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher
  * Python 3.6

GPU availability is recommended to train the models. Otherwise, set the use_cuda flag in parameters.yaml to false.

### Requirements

- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)


## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file ./experiments/parameters_example.yaml
```

Most of the parameters in the yaml configuration file are self-explanatory. 
The 'experiments' folder contains config file for the results reported in the [DEFT paper](https://arxiv.org/abs/2211.11979).

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch. For example:
```sh
python log_analyzer.py log/filename.log
```


## Reference

[1] Anson Bastos, Abhishek Nadgeri, Kuldeep Singh, Toyotaro Suzumura, Manish Singh. [Learnable Spectral Wavelets on Dynamic Graphs to Capture Global Interactions](https://arxiv.org/abs/2211.11979). AAAI 2023.

## BibTeX entry

If you use our work kindly consider citing:


```
@misc{https://doi.org/10.48550/arxiv.2211.11979,
  doi = {10.48550/ARXIV.2211.11979},
  url = {https://arxiv.org/abs/2211.11979},
  author = {Bastos, Anson and Nadgeri, Abhishek and Singh, Kuldeep and Suzumura, Toyotaro and Singh, Manish},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learnable Spectral Wavelets on Dynamic Graphs to Capture Global Interactions},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Acknowledgements
This code has been adapted from [EvolveGCN](https://arxiv.org/abs/1902.10191). Many thanks to the authors for sharing the code.
