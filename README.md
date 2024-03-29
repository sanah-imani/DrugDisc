# DrugDisc



Computational strategies have been used to expedite and reduce cost of the process of drug discovery (1) by identification of lead compounds that have strong affinity for their respective target molecules. Currently, high-throughput techniques are been used to screen drugs. The order of magnitude of the drugs that can be tested in a lab through this method is 106 (2). This means that the time for drug searching becomes really large and inefficient. In fact, the current drug-discovery process can take up to 12–16 years of immense research, big financial investments, and pre-clinical and clinical trials before a molecule can be chosen as a drug (3). In this project, three elaborate stages have been computationally represented: the generation of novel molecules, the prediction and calculation of molecular properties, and the classification of these molecules as active or not active with respect to target molecules. Each of these stages was accomplished by the use of a Long-Short Term Memory model auto-encoders built through Keras and Tensorflow (4)(5), Rdkit libraries, and Random Forest (6) respectively (multiple other models were tested too). The data was imported from ChemBL and consists of active drugs for lung carcinomas, and to check the validity of the models the data was split into train data and test data. The hybrid model will be used in generation of drug-like compounds and then the output will be filtered to select compounds that can chemically exist using an Rdkit Filter and molecules will be sanitized using the same software.(7) Next, Rdkit will be utilized in order to predict the value of some chemical properties. These will be input to the Random Forest in order to predict if the drug-like compound is a potential candidate against the lung carcinoma. The ligand-protein binding affinity can be also predicted using docking programs (AutoDock Vina) to provide affinity values for different conformations of the protein and the target molecule (various target molecules found in the ChemBL database will be sampled) (8). Additionally, a potential use of an evolutionary algorithm has been experimented in the extension. This algorithm is usually used with standard neural networks in drug compound identification (9). In this in silico study, the evolutionary algorithm will be used to make a more robust drug activity prediction model.

The file containing all smiles generated has not been attached due to size.


![image](https://user-images.githubusercontent.com/29833463/148157181-fe3ba7c0-3150-4bd4-8390-8fd80b09eb53.png)

<b> Long Short Term Memory (LSTM) autoencoder </b> 

<ol>
  <li> Type of Recurrent Neural Network that have feedback loops allowing information to persist within the system. </li>
  <li> Captures long-term dependencies: Samples the categorical distribution of the next object </li>
  <li> Input Sequence Data -> Compressed representation </li>
</ol>

![image](https://user-images.githubusercontent.com/29833463/148158380-5301a436-2528-4c95-a223-f5c0d7bfac44.png)

<b> Results </b>

Model trained for approximately 100 epochs. Produced around 150 valid SMILES.

Used rdkit sanitization methods to isolate sensible compounds and making modifications using chemical statistics.

Sampled 10 random compounds.
