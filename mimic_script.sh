
MIMIC_PATH=$1
echo "Reading mimic data from $MIMIC_PATH"
python get_mimic_data.py $MIMIC_PATH/00 examples/mimic_fields.txt> patients.xml

echo "Downloading ICD9 codes"
cd examples/ICD9
python load_ICD9_structure.py

cd ../..
echo "Building ICD9 codes"
python build_structured_rep.py code examples/ICD9/code
ls Structures

echo "Preprocessing patients"
python preprocess_patients.py 1000 patients.xml examples/ICD9/settings.xml


