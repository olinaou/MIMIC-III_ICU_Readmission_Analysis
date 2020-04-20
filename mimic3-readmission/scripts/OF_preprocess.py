import os
from subprocess import call

base = "../../../mimic-iii-clinical-database-1.4/"
input_data = "cut1000"
subject_data = "output_subjects_1000"
episodes_data = "output_episodes_1000"
listfiles_data = "output_listfiles_1000"

# works in windows but not mac:
# os.system("extract_subjects.py " + base + input_data + " " + subject_data)
# os.system("validate_events.py " + subject_data)
# os.system("create_readmission.py " + subject_data + " " + "--custom-model model-1")
os.system("extract_episodes_from_subjects.py " + subject_data)
os.system("create_readmission_data.py " + subject_data + " " + episodes_data)
os.system("split_train_val_test.py " + episodes_data + " " + listfiles_data)

# call(['python3', 'extract_subjects.py', base + input_data, subject_data])
# call(['python3', 'validate_events.py', subject_data])
# call(['python3', 'create_readmission.py', subject_data])
# call(['python3', 'extract_episodes_from_subjects.py', subject_data])
# call(['python3', 'create_readmission_data.py', subject_data, episodes_data])
# call(['python3', 'split_train_val_test.py', episodes_data, listfiles_data])


