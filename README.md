# Project Description

This project aims to develop a health status prediction system. The first step consists on the development of a fuzzy knowledge graph>
The second step consists on training this graph on four graph neural networks algorithms. 

  # Dataset 

  The used Dataset was taken form the follwoing [link](https://www.kaggle.com/datasets/manideepreddy966/wearables-dataset) 

  # Ontology 

  The Ontology folder contains the Fuzzy HealthIoT ontology 

  # Fuzzy Knowledge graph Generation

      Contains the code to create a fuzzy Knwledge grap based on the dataset and the fuzzy ontology. To execute this code: 
          1. Clone the folder and create a virtual environment 
          2. Install the necessary package in the requirements.txt file: <pip install -r requirements.txt>
          3. execute main.py: python main.py
  # Health Status prediction

    In this step we train the fuzzy knowledge graph using four differents graph neural networks algorithms (GNNs) namely: GraphSAGE, Graph convolutional networks, Graph attention networks, Graph isomorphism netwro
    The notebook file contains the code that change the RDF graph to a graph format where the GNNs algorithms can be trained, and the training code in the four algorithms and the the evaluation
    
