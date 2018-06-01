# MIMIC-III_ICU_Readmission_Analysis
This is the source code for the paper 'Analysis and Prediction of Unplanned Intensive Care Unit Readmission'

## Requirements

1. You need to first get access to MIMIC-III data by  yourself from https://mimic.physionet.org/, and save all of the csv files into a directory.
We do not provide the MIMIC-III data itself. 

2. Install the following packages:

- numpy
- pandas
- Keras
- scikit_learn


3. Download pre-trained ICD 9 Embeddings
In this study, our models were trained on a lower dimension embedding of ICD9. We apply the pretrained 300-dim embedding for each ICD9 code. You can download one of the following disease embedding:
- [claims_codes_hs_300.txt] (https://github.com/clinicalml/embeddings) : 300 dimensional ICD_9 embeddings
Please download the `claims_codes_hs_300.txt.gz`, extract it, and put the `claims_codes_hs_300.txt` into `/embeddings` folder.


## Preprocessing
1. Add the path to the `PYTHONPATH` (sorry for this).

    export PYTHONPATH=$PYTHONPATH:[PATH TO THIS REPOSITORY]


2. For each patient, `SUBJECT_ID`, we generate the `stays.csv`, `events.csv`, `diagnoses.csv`,`transfers.csv`,`procedures.csv`and `prescriptions.csv`. We place them into the directory `data/[SUBJECT_ID/`. In this study, we only use `stays.csv`, `events.csv`, and `diagnoses.csv`. The rest of the file will be used in the future work.

       python scripts/extract_subjects.py [MIMIC-III CSVs PATH] [OUTPUT PATH]

3. We then try to to fix some missing data issue(ICU stay ID is missing). If the issues cannot be solved, we then removes the events. The code is provided by [].


       python scripts/validate_events.py [OUTPUT PATH]
       
4. We then preprocess the label of readmission, The following 4 cases of readmission are computed in this steps. The output will be saved in `stays_readmission.csv`.
(1) The patients were transferred to low-level wards from ICU, but returned to ICU again.
(2) The patients were transferred to low-level wards from ICU, and died later.
(3) The patients were discharged, but returned to the hospital within the next 30 days.
(4) The patients were discharged, and died within the next 30 days. 

       python scripts/create_readmission.py [OUTPUT PATH]


5. We then divide the `stays_readmission.csv` into separate ICU stays. We use the following commands to extract the time series information from chat events, `events.csv`, and saved in `episode{#}_timeseries_readmission.csv`. The demographic information will be saved in `episode{#}_readmission.csv`.

       python scripts/extract_episodes_from_subjects.py [OUTPUT PATH]

6. We then generate the readmission datasets that will be used in the following experiments. We also remove all cases that patients who died in ICU int his steps. There will be a file `listfile.csv` including all ICU stays that will be used in this study.

       python scripts/create_readmission_data.py [OUTPUT PATH] [OUTPUT PATH 2]

7. In this steps, we then split the processed patients into training 80%, validation 10% and testing 10% partitions and conduct a five-fold cross validation. Note that one patient may have multiple records, so the number of items may not equal in each fold. Before executing the following commend, please edit the directory in the code to the directory of `listfile.csv`.

       python scripts/split_train_val_test.py

## Executing models
In this section, we use LSTM and LSTM_CNN as examples. you may want to different models in `common_keras_models`.

1. Baselines. To run the baseline models (e.g., LR,NB,RF,SVM), please edit the directory in the code to the corresponding directory.

       cd /mimic3models/readmission_baselines/logistic_cv_0
       python svm_s_p.py
	
2. LSTM F48-h CE + ICD9. In this steps, we use the first 48 hours chart events after admitting to IUC to predict readmission.

       cd /mimic3models/readmission_f48/
       python3 -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 

3. LSTM L48-h CE + ICD9. In this steps, we use the last 48 hours chart events before discharging from IUC to predict readmission.

       cd /mimic3models/readmission_no_d/
       python3 -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 

4. LSTM L48-h CE. In this steps, in order to see the impact of disease features, we remove the ICD 9 imbedding features to predict readmission.

       cd /mimic3models/readmission_no_icd9/
       python3 -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 

5. LSTM+CNN L48-h CE + ICD9 + D. In this steps, we include all of the information that we preprocess to predict readmission.
	
       cd /mimic3models/readmission/
       python3 -u main.py --network ../common_keras_models/lstm_cnn.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 

References
[1] https://github.com/YerevaNN/mimic3-benchmarks
[2] https://github.com/clinicalml/embeddings
