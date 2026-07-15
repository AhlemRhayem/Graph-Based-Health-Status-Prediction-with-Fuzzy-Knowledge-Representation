# Project Description

This project aims to develop a health status prediction system. The first step consists on the development of a fuzzy knowledge graph>
The second step consists on training this graph on four graph neural networks algorithms. 

  # Dataset 

  The used Dataset was taken form the follwoing [link](https://www.kaggle.com/datasets/manideepreddy966/wearables-dataset) 
**
  Description of the Original dataset **

In our study we used the Personal Health Data dataset sourced from Kaggle\footnote{https://www.kaggle.com/datasets/manideepreddy966/wearables-dataset}. The dataset originally consists of 10,000 patient records and 34 features, where each row represents an individual patient and each column corresponds to a specific medical, demographic, or lifestyle attribute. The variables include numerical features such as age, weight, height, heart rate, blood oxygen level, sleep duration, and body fat percentage, as well as categorical features such as gender, smoking status, alcohol consumption, medical conditions, and mood. The primary target variable is the HealthScore of the patient. 
A critical examination of the original dataset revealed that, while it is presented as a health-related dataset, it was synthetically generated without adherence to real-world medical or epidemiological distributions. This was evidenced by near-perfectly uniform distributions across all categorical variables (e.g., 50% smokers vs. 50% non-smokers; 33% male, 33% female, 33% other), which are statistically implausible and inconsistent with established population health statistics. Furthermore, the string label "None" used for healthy patients was silently converted to missing values (NaN) upon data loading, effectively rendering the healthy cohort invisible in any analysis. Numerical variables such as blood oxygen level (SpO₂) exhibited an artificial spike at 100%, and body weight extended to physiologically extreme values, both of which are inconsistent with real clinical measurements.
<img width="945" height="689" alt="image" src="https://github.com/user-attachments/assets/3ce85713-166a-46c5-b353-3107fd26cb72" />

**Dataset Reconstruction**
To address the mentioned issues, the dataset was reconstructed to reflect real-world distributions grounded in published epidemiological references. Specifically:
1.	Gender was set to a balanced distribution of approximately 49% male and 49% female, with 2% identifying as other, in line with current demographic trends. 
2.	Smoking prevalence was set to 22%, consistent with WHO Global Tobacco Report estimates. 
3.	Alcohol consumption was redistributed to 43% non-drinkers, 43% moderate drinkers, and 14% heavy drinkers, following WHO global alcohol reports and related literature. 
4.	Medical condition prevalence was grounded in CDC and WHO data, with approximately 51% healthy, 32% hypertension, 11% diabetes, and 6% comorbid hypertension and diabetes. 
5.	Condition assignment was made age-dependent, such that younger patients were predominantly healthy and chronic conditions increased with age, reflecting known epidemiological patterns. 
6.	Blood oxygen saturation (SpO₂) was capped at 99% with a clinically appropriate mean of 97.8% and a standard deviation of 1.0. 
7.	Body weight was derived from a realistic BMI distribution capped at 40 kg/m². 
Correlated variables such as heart rate, ECG results, medication use, and health score were also adjusted to reflect plausible clinical associations with the corrected features. The use of synthetic data is justified by patient privacy constraints, and the corrected generation process is grounded in documented real-world distributions, addressing a key limitation of the original dataset.

**Exploratory Data Analysis**
The exploratory analysis of the corrected dataset revealed several meaningful and epidemiologically coherent insights, as summarised in Figure 1. The gender distribution is balanced between male and female, which minimises bias in model predictions (Figure 1a). The distribution of medical conditions reflects real-world prevalence: the majority of patients (52.4%) are healthy, while hypertension (29.6%) and diabetes (11.5%) are the most common conditions, with a smaller comorbid group (6.5%), as shown in Figure 1c. A clear age gradient was observed across conditions (Figure 1b), with healthy patients being notably younger (mean age approximately 38 years) and the comorbid hypertension and diabetes group being the oldest (mean age approximately 61 years), consistent with the known epidemiology of chronic disease onset. Analysis of lifestyle factors confirmed expected associations (Figure 1d): smoking and heavy alcohol consumption were both positively associated with hypertension and diabetes, while the healthy cohort exhibited the lowest rates of both behaviours. Heart rate, blood oxygen saturation (SpO₂), and health score all showed realistic, clinically interpretable distributions, lending credibility to downstream modelling based on this corrected dataset.

<img width="835" height="683" alt="image" src="https://github.com/user-attachments/assets/01edbdfa-52ac-4ad8-afb9-4d00073d23ab" />


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
    
