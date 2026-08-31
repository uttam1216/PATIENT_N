# PATIENT+N
PATIENT+N: Profiling feATure Importance for focal Epileptic seizure oNset detecTion for New Patients

<b>Disclaimer: </b> This software is only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market. If you use the source code or any part of our research in any publication, please cite our research - https://doi.org/10.1007/978-3-032-37664-0_27

<b>Notes: </b> <br>
1) To access the publicly available TUH Seizure (TUSZ) data used in this research, a Non-Disclosure Agreement (NDA) needs to be signed with the dataset author: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/   <br>
In patient_n_dummy_data.csv we only give an example dummy data format that model needs to run with.  <br>
2) Given one has the data access, by following the below instructions, the codes provided can be run to reproduce results depicted in the submitted paper. It is recommended to run the python commands sequentially in a tmux session, as some feature extraction may take hours.<br>

<b>Overview: </b> The project follows a multi-stage pipeline consisting of feature extraction, patient-specific modeling, clustering of seizure mechanisms, and multimodal seizure detection. Intermediate artifacts such as embeddings, clustering outputs, and trained models are stored in dedicated directories under src to ensure reproducibility and clear experiment tracking. The final model integrates tabular EEG features and scalogram representations through a cluster alignemnt mechanism, enabling patient-adaptive seizure onset detection. <br>

<b> Focal Seizure Onset: </b>
In focal seizure it can happen that one of the annotated bipolar channels e.g. 'FP2-F8' starts a seizure at start_time 32 sec, followed by other bipolar channels e.g. F8-T4 starting later e.g. 41st second. However, to still detect seizures early, we set a hard limit of <b>20 seconds</b> after start_time of the first annotated seizure channel for a given session as the seizure-onset time period. <b> Our EEG segment windows (8 seconds long), move only within this 20 second zone </b> and NOT over the entire annotated seizure time period. The EEG segment is considered valid and saved for seizure-onset class only if it contains at least one channel whose annotated seizure start time falls within this EEG segment.

<b>Project Directory Structure: </b> When the listed programs are run sequentially by just providing a source folder src as argument, all required folders will get created with meaningful names in a structure as shared below. All intermediate data, trained models, embeddings, clustering outputs, and diagnostic results are stored under this root directory src. The project follows a 5-fold cross-validation setup, where each fold contains patient-wise train, validation, and test splits. <br>

```
src/
│
├── fold_{1|2|3|4|5}/
│   ├── train_pat.pkl
│   ├── val_pat.pkl
│   └── test_pat.pkl
│
├── XGBoost_SHAP/
│   └── fold_{1|2|3|4|5}/
│       ├── pat_id_wise_data/
│       ├── cluster_centroids_category_space.csv
│       ├── cluster_category_weights_normalized_fold{fold}.csv
│       ├── elbow_curve_kmeans.png
│       ├── patient_to_cluster_setA.json
│       ├── patient_to_cluster_setB.json
│       ├── setA_patient_xgb_summary.csv
│       └── cluster_viz/
│
├── scalograms/
|    ├── sz/
|    └── ns/
│
├── efficientNetB5/
│   └── fold_{1|2|3|4|5}/
│       ├── best_model.pt
│       ├── config.json
│       └── train_log.csv
│
├── iter2_emb/
│   └── fold_{1|2|3|4|5}/
│       ├── scalo_emb/
│       │   └── {train|val|test}/{sz|ns}/{pstrst}.npy
│       ├── tabular_window_emb/
│       │   └── {train|val|test}/{pstrst}.npy
│       └── patient_mean_emb/
│
├── runs_cluster_gated/
│   └── <run_name>/
│       ├── config.json
│       ├── cv_summary.json
│       ├── folds/
│       ├── diagnostics/
│       └── figures/
│
├── optuna_tmp/
   ├── <study_name>.db
   ├── best_params_final.json
   ├── optuna_trials.csv
   └── sensitivity_plots/

```

<b> Folder Descriptions: </b>  <br>
<b> fold_{1|2|3|4|5}/ </b>  <br>
Contains the patient-wise dataset splits used in cross-validation. <br>
  + train_pat.pkl – training set dataframe <br>
  + val_pat.pkl – validation set dataframe <br>
  + test_pat.pkl – test set dataframe <br>
Each split ensures no patient overlap between sets. <br>

<b> XGBoost_SHAP/ </b> <br>
Stores the outputs of patient-specific XGBoost modeling and SHAP analysis, used to derive feature-category importance and cluster patients based on seizure mechanisms. Important files include: <br>
  + cluster_centroids_category_space.csv – cluster centroids in feature-category space
  + cluster_category_weights_normalized_fold{fold}.csv – normalized feature-category weights per cluster
  + elbow_curve_kmeans.png – elbow plot used to determine the number of clusters
  + patient_to_cluster_setA.json – mapping of patients to clusters
  + setA_patient_xgb_summary.csv – summary of patient-level model performance

<b> scalograms/ </b> <br>
Stores precomputed scalogram representations of EEG segments: 
  + sz/ – seizure windows
  + ns/ – non-seizure windows
These are used as inputs for the scalogram branch of the multimodal model <br>

<b> efficientNetB5/ </b> <br>
Contains trained EfficientNetB5 models used for scalogram-based feature extraction. Each fold stores:
  + best_model.pt – trained model checkpoint
  + config.json – model configuration
  + train_log.csv – training statistics

<b> iter2_emb/ </b> <br>
Stores intermediate embeddings extracted from EEG windows. Subfolders include:
  + scalo_emb/ – scalogram embeddings for seizure (sz) and non-seizure (ns) windows
  + tabular_window_emb/ – tabular embeddings derived from engineered features
  + patient_mean_emb/ – patient-level aggregated embeddings

<b> runs_cluster_gated/ </b> <br>
Stores the outputs of the cluster-gated multimodal training pipeline, including:
  + configuration files
  + cross-validation results
  + fold-wise model checkpoints
  + cluster alignment diagnostics
Each run is saved in a timestamped folder <br>

<b> optuna_tmp/ </b> <br>
Contains artifacts from Optuna-based hyperparameter optimization, including: 
  + the Optuna study database
  + best hyperparameters
  + trial logs
  + hyperparameter sensitivity plots



<b> Repository Usage: </b> <br>
All python packages can be installed by running the following command in the terminal: <br>
<b> <i> pip install -r requirements.txt </i> </b> <br>
Additionally requirements_exact.txt is also provided for reproducibility, that has all libraries of the environment in which all my code ran, in case one has issue with the minimum required packages listed in requirements.txt

<b> Loading TUH  Seizure Annotations: </b> <br>
The following python command will load all seizure annotations needed for data processing in a postgresql database (install if not existing) for easy and faster data processing later:
<b> <i> python3 load_tuh_metadata_sqlalchemy.py tuh_eeg_sz_filepath  </i> </b> <br>
where an example format of tuh_eeg_sz_filepath (as present in my disk after download from the TUH dataset authors) is: "/media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*"

<b> Data Preprocessing: </b> <br> Once you have the TUH EEG Seizure corpus along with the seizure annotated times present in the freshly created postgresql db achieved via previous script run, by running the following command, eeg segments for seizure-onsets and non-seizure gets created at a location eeg_filepath <br>
<b> <i> python3 generate_fnsz_eeg_segments.py eeg_filepath </i> </b> <br>
where eeg_filepath is of format src+'eeg_segments', src='/media/data/ukumar/iBehave/data_files/feb25/' or wherever you want to make a parent folder for all processes; additionally the length of eeg segments required (in seconds) and the stride for moving window (in seconds) can be given along after the input_eeg_filepath argument. By default these values are set to have 8 sec and 1 sec respectively. <br>
After this we normalize the generated eeg_segments on per-channel per-patient basis using the following command giving a location for saving the normalized segments. <br> 
<b> <i> python3 normalize_eeg_segments.py eeg_filepath </i> </b> <br>
where eeg_filepath is of format src+'eeg_segments'. This saves the normalized eeg segments in same parent folder src, where the earlier eeg_segments existed with the name of folder as normalized_eeg_segments and is now ready for all types of feature extraction <br>

<b> Feature Extraction: </b> <br> Once you have the normalized eeg segments saved in a folder path normalized_eeg_segments_path achieved from above step, then respective image and numeric features can be extracted from it by running the following commands in your terminal: <br>
<b> <i> python3 generate_scalograms.py  normalized_eeg_segments_path </i> </b> <br>
where normalized_eeg_segments_path is of format /media/data/ukumar/iBehave/data_files/feb25/normalized_eeg_segments. This will generate the required image features and save it in same parent folder as the normalized_eeg_segments andwill name the newly created folder as scalograms. Next we also want to have numeric combined features which can be achieved by running teh following command. <br>
<b> <i> python3 feature_processing.py normalized_eeg_segments_path </i> </b> <br>
where normalized_eeg_segments_path is src+'normalized_eeg_segments'; running this command will save all the numeric features in three files train_pat.pkl, val_pat.pkl and test_pat.pkl inside each fold_{fold_num} folder inside the main parent folder src <br> 

<b> Scalogram Embeddings: </b> <br> 
Once you have normalized pkl files corresponding to the EEG segments for each fold saved in folder root folder path src where src has each fold respective files, we can run the following to finetune efficientNet B5 model on the scalograms and save them for use by the end-to-end PATIENT_N model. Exceuting the below line will create a folder efficientNetB5 inside src that will save teh best models for each folds which are then used by same program to create teh scalogram embeddings and is saved inside src+'iter2_emb/fold_{fold_num}/scalo_emb/{train|test|val}/{sz|ns}' <br>
<b> <i> python3 train_efficientNet_and_create_scalogram_embeddings.py scalograms_path </i> </b> <br>
where scalograms_path = src+'scalograms'

<b> Patient Specific Modelling and Cluster Profile Creation: </b> <br> Once you have all numeric features corressponding to the normalized pkl files corresponding to the EEG segments for each fold saved in folder root folder path src where src has each fold respective files, we can run the following to run xgboost model for each patient and get cluster ids assigned to the originals datasets by running the following with src as argument. All train, test and val .pkls will be picked up from fold_{fold_num}/ folder of each fold and the output files will be created inside the folder src + XGBoost_SHAP/fold_{fold_num}/ <br>
<b> <i> patient_specific_modelling_and_cluster_profiling.py src </i> </b> <br> 

<b> Focal Seizure-Onset Detection Model: </b> <br> Once you have the features processed, the model can be run by executing the following command in the terminal: <br>
<b> <i> python3 model.py src </i> </b> <br>
A jupyternotebook of this final end-to-end model is also shared inside code_files

<b>Affiliations: </b> The source code is being published in affiliation with University of Bonn (https://www.uni-bonn.de/de) and the Lamarr Institute (https://lamarr-institute.org/). This work was partially funded by the Ministry of Culture and Science of the State of North Rhine-Westphalia, Germany (“iBehave”) and the German Federal Ministry of Research, Technology and Space (BMFTR) (“SecureNeuroAI”, 16KIS2345K). If you use the source code or any part of our research in any publication, please cite our research - https://doi.org/10.1007/978-3-032-37664-0_27

<b>Acknowledgements: </b> This work was partially funded by the Ministry of Culture and Science of the State of North Rhine-Westphalia, Germany (“iBehave”) and the German Federal Ministry of Research, Technology and Space (BMFTR) (“SecureNeuroAI”, 16KIS2345K). 




