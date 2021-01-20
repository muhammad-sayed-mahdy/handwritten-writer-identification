# handwritten-writer-identification
handwritten writer identification using classical machine learning pattern recognition techniques 

## Requirements
you can install the requirements using  
`pip install -r requirements.txt`  
or you can use conda to create a virtual environment and install packages using  
`conda env create -f environment.yaml` 

## Sample result
![sample test](https://user-images.githubusercontent.com/32793798/105159229-82193e80-5b17-11eb-808e-92079282be79.png)

At the right is the test image.  
At the left are the tuning images of the 3 authors sorted from top to bottom (1, 2, 3).  
Our system predicted author 1 for this test sample.

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
    3- run: `python src/main.py`
    4- check results at `results.txt` and timing at `time.txt`

A test case folder in the data folder should contain a folder for each author containing 2 images of that author, and a test image.  
Results file will contain -for each test case at the `data` folder- the predicted number of author that wrote the `test.png` image.  
Author number corresponds to the folder number that contains the images of the author.
        
## Statistics
We used [IAM Handwritten Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) to validate our pipeline.  
It has the following statistics.

    + Total number of authors: 657
    + Total number of authors with three or more forms: 159
    + Total number of forms that correspond to the 159 authors that have 3 or more forms: 899
    + Total number of forms in the system: 1539

We ran our pipeline on the 159 authors that have three forms or more, selecting each form of one author at a time as a test image, and retrieving two other random authors for tuning, each author of the 3 has 2 tuning images.  
We achieved **99.44 %** accuracy, failing in only 5 trials out of the 899.

## How it works
The pipeline is as follows
0. read the tune images and test image
1. preprocess images and segment lines out of the image, then apply otsu binary thresholding.
2. perform feature extraction using [wavelet transform](https://ieeexplore.ieee.org/document/5597888), then CSLBCoP that is local binary pattern (LBP) followed by grayscale co-occurrence matrix (GLCM) proposed by this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045790617322401).
3. perform principal component analysis (PCA) to extract useful features from the resultant feature vector.
4. fit the classifiers (we used SVM and KNN) to the tuning images, and predict the test image
5. take a vote from the classifiers (in case SVM predicted with low confidence)
6. report the predicted author and the time taken by pipeline without IO operations.

For more information, you can refer to the [document](doc/Pattern%20Report.pdf)

## File Structure
1. `src` folder contains the main code for preprocessing, feature extraction, classification, pipeline, and evaluation.
2. `graphs` folder contains the useful generated graphs during development.
3. `doc` folder contains the project document.
4. `script` folder contains a shell script to organize the tuning data.

### script folder
you can use the `script` folder to organize the files of the IAM database, just move all the complete form images (that are like the test sample, not the words nor lines images) of the IAM database and put it in `script` folder.  
Running the script will create a `data_tune` folder at the project directory and put the images of the same author at a separate folder in it.