# Ner Training For Address Entities

You can use this repository for ner training. If you have json file, you need to convert it to HuggingFace dataset by using `dataset_convertor.py` and save it.
After saving dataset to your local, all you need is to setup your experiment to `run_ner.sh`


## Usage
```bash
# Setup virtual environment
virtualenv venv && source venv/bin/activate && pip install -r requirements.txt

# Edit ./run_ner.sh for other configurations before running

chmod +x run_ner.sh # Make the file executable if it's not already
./run_ner.sh # Run
```
