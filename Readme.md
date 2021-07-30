**COMP5423 NATURAL LANGUAGE PROCESSING**

**Lab1 Homework: Emotion Classification**

# File structure 
The programm contain several files, the following are basic introduce and discrible the files.   
- /Css
  - style.css ------ a css file of the template of webpage 
- /templates
    - classification.html ------ a html file of the structure of the webpage, running at the endpoint localhost:5000/
- app.py ------ a python file to run emotion classification in web application  
- Model.py ------ a python file to run emotion classification in python terminal that output accuracy score and test data result 
- test_data.txt ------ a txt file contain testing data 
- test_prediction.txt ------ a txt file output by model.py that contain all testing data result   
- train.txt ------ a txt file contain all training test data
- val.txt ------ a txt filr contain all validation testing data 
- lab 1 repot ------ a pdf file for lab 1 report

# Requirement Environment
- Visual Studio Code (version 1.53)
- Python 3.5 - 3.8
- Requires the pip
    - pip install pandas
    - pip install numpy
    - pip install nltk
    - pip install -U scikit-learn
    - pip install Flask

# How to run it 
The programme have two parts, 
- i) web application, 
- ii) test_prediction data and accuracy score result

For part i,

1. Run app.py in Visual Studio Code
2. open browser and visit http://localhost:5000/
3. Input text in the textarea in the webpage 
4. Click submit
5. Result a emoji 

For part ii, 

1. Run Model.py in Visual Studio Code
2. Output the result of accuracy score
3. Output a file 'test_prediction.txt'

