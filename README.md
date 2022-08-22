# Internship_2022_Fellay_LAB

Within the data folder:
- GSE172114_important_genes__normalised_count.xlsx
contains the normalized read counts of the 200 DEGs from 65 samples verifying:
padj_value < 0.05/9078
| L2FC | > 1.5
to which a label has been added to differentiate samples from healthy patients from those with a critical form of COVID19.
 - all_normalised_gene_counts.xlsx 
 contains the normalized read counts of the 37194 genes from 65 samples verifying:
 total number of reads across the samples larger > 5 (post filtred 47752 genes  )
 
Within the script_data folder:
- all_count_data : 
 scripts dedicated to the large gene count data set from 65 samples (different patients) and counting the reads of 37194 genes (not only the DEGs)
 - small_count_data : 
 scripts dedicated to the reduced gene count data set from 65 samples (different patients) and counting the reads of 200 genes (not only the DEGs)
 
- Data_processed_for_classification.py /  All_counts_data_processed_for_classification.py :
Load the normalized read counts of the 200 / 37194 DEGs / genes  processed by DESeq2, 
Separats the response variable (label) from the rest of the data  
One hot encode the response variable 
Summarizes how much samples , genes and patient groups have been iodentified to allow the user to verify the accuracy of this step.


WORKFLOW  : all_gene_count case
1 / The user needs first to run for each model , the scripts within the model selection folder . 
The results of the run will be printed in a txt file within the hyper_param_tunning folder under ML_results/all_count_data . 
In the latter there will be the results of the 10 external layers of the two layer cross validation
here is an output example :
  Baseline model accuracy 0.67
  0.8939393939393939 {'C': 0.01, 'solver': 'newton-cg'}
We can first see the accuracy of the baseline model, followed by that of the tunned one and the hyperparameters combinaison that it possible to achieve this accuracy.
The user can then either retain the combination that comes up the most often or the one that has allowed tyo achieve the highest accuracy.

2/ The user will then have to change the hyperparameters of this same model within the script under script_data/scripts/all_count_data/Performance_evaluation/ . 
The scripts ,as its name suggests allows , to evaluate the classification of the tuned model . 
The results of the run will be printed under ML_results/all_gene_count/classification_performance so each model will have four outputs:
the confusion matrix / the value of the accuracy and the recall (sensitivity), the roc curve and a report with different performance metrics such as the f1 score .

BEST PRACTISE : 
The user can also finish tunn all the models than edit the hyper parameters of these under the performance evaluation scripts with the chosen ones . 
than run the classification_perf_summary script, that generates the output discussed above for all the models in parallel and print a table with the relevant classification metrics for each classifier facilitating their comparaison . 

3/the scripts under script_data/scripts/all_count_data/RFECV, Allows the user to carry out a recursive feature election with a cross validation . 
It's an accurate way to achieve a feature selection and allows to identify how many genes each tunned model uses in order to reach the highest classification accuracy 
this script allows to extract within a txt file under ML_results/all_gene_count/feature_importance the names of the most relevant genes as well as a plot with the first 50 ones  .

The file text can then be used to carry out an enrichment analysis (STRING-db) and thus check whether there are pathways that are enriched or to identify genes that are expected to  relevant.

4 / If the user would like to compare pairwise the optimised classifiers in an correct way, based on their misclassification rate, he/she can use the scripts under script_data/scripts/all_count_data/pairwise_comparaison/

- models_baseline_compare.py : performs a correlated t-test to assess whether the classification accuracy of each of the tuned models is significantly (confidence threshold of 0.05) better than that of the baseline model

- model_pairwise_comparaison.py : performs a correlated t-test to assess whether the classification accuracy of one tuned models is significantly (confidence threshold of 0.05) better than that of another one  . The script goes through each pair 
