# handwritten-writer-identification
handwritten writer identification using classical machine learning pattern recognition techniques 

### Requirements
you can install the requirements using  
`pip install -r requirements.txt`  
or you can use conda to create a virtual environment and install packages using  
`conda env create -f environment.yaml` 

### STAT
    + Total number of authors: 657
    + Total number of authors with three or more forms: 159
    + Total number of forms in the system: 1539

## How to Run:
    1- Make sure to install all of the requirements.
    2- Place test data at folder named data to be ordered in that matter:
        data/001/1/1.png
        data/001/1/2.png
        data/001/2/1.png
        data/001/2/2.png
        data/001/3/1.png
        data/001/3/2.png
        data/001/test.png
    and so on...
    3- run: python src/main.py
    4- check results at results.txt and timing at time.txt
        