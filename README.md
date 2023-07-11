# graph-coloring-gnn

This is a repository made for the exam of Advanced Machine Learning for Physics of the course of MSc. in Physics at the Sapienza Universiy of Rome. The purpose of the notebook is recreate the results obtained in the article: "Graph Coloring with Physics-Inspired Graph Neural Networks" (https://doi.org/10.48550/arXiv.2202.01606) and should you use any of the methods presented here it should be done by citing their article:

```latex
@article{Schuetz_2022,
	doi = {10.1103/physrevresearch.4.043131},
  
	url = {https://doi.org/10.1103%2Fphysrevresearch.4.043131},
  
	year = 2022,
	month = {nov},
  
	publisher = {American Physical Society ({APS})},
  
	volume = {4},
  
	number = {4},
  
	author = {Martin J. A. Schuetz and J. Kyle Brubaker and Zhihuai Zhu and Helmut G. Katzgraber},
  
	title = {Graph coloring with physics-inspired graph neural networks},
  
	journal = {Physical Review Research}
}
```

The data used to train the GNN models is not included in the repository and can be downloaded within the notebook as follows:
```python
import os

# Here we create a directory for storing the input data
input_data_path = './data/input/COLOR/instances'
if not os.path.exists(input_data_path):
    os.makedirs(input_data_path)

#Here we download the input data
! wget https://mat.tepper.cmu.edu/COLOR/instances/instances.tar -P ./data/input/COLOR/

#Here we extract the input data
! tar -xvf ./data/input/COLOR/instances.tar -C './data/input/COLOR/instances'

```
