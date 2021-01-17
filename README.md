# handwritten-writer-identification
handwritten writer identification using classical machine learning pattern recognition techniques 

### Requirements
    ~ pip install -r requirement.txt 

### STAT
    + Total number of authors: 657
    + Total number of authors with three or more forms: 159
    + Total number of forms in the system: 1539

## Tuning

1- regz with svm
    + ran for 100 epochs
    + tested with (3,2,1) forms (ie. three authors with two forms each for fitting and one for testing)
    + results at graphs/tune_regz_svm.png