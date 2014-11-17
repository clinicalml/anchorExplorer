
MIMIC_PATH=$1
python get_mimic_data.py $MIMIC_PATH/00 examples/mimic_fields.txt> patients.xml

cd examples/ICD9
python load_ICD9_structure.py

cd ../..
python build_structured_rep.py code examples/ICD9/code
ls Structures

python preprocess_patients.py 1000 patients.xml examples/ICD9/settings.xml


