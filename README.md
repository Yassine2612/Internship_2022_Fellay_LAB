# Internship_2022_Fellay_LAB

Within the data folder:
- csv file: GSE172114_important_genes__normalised_count.xlsx
which contains the normalized counts of the 200 DEGs verifying:
padj_value < 0.05/9078
| L2FC | > 1.5
to which a label has been added to differentiate samples from healthy patients from those with a critical form of COVID19.

Within the scripts folder:
- Data_processed_for_classification.py:
Load the normalized counts of the DEGs revealed by DESeq2, 
Separats the response variable (label) from the rest of the data  
One hot encode the response variable 
Summarizes how much samples , genes and patient groups have been iodentified to allow the user to verify the accuracy of this step.

- Optimized_model_selection.py: 
Creates five different classification models : Regularised logistic regression, RandomForest, Support vector Machine, XGB and Multilayer Perceptron using functions sklearn library.
Performes model selection (hyper-parameter tunnig ) in a two layer K-fold cross validation
Evaluates classification performance in a one layer hold out (20% testing) cross validation . Outputs : accuracy , f-1 score , precision , recall ...
Reveals and plot the attributes (genes) contributing the most to the decision taken by the classifier (feature selection)

Ensembl_classifier.py: 
Groups the five classifiers tuned in the previous within an Ensemble voting classifier (soft) , 
Perform a model selection using a two layer cross validationin order to determine the optimal subset to include to maximize classification accuracy and sensitivity
Evaluate the performance of the EVC  using a hold-out CV and extracts the attributes that affects the most the decision.

Within the toolbox_02450 folder
Auxiliary scripts provids utility functions
