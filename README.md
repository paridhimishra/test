Running Project

Download and uncompress the project file culture-amp.7z
To run as python project use virtual environment (recommended as this install nltk and gensim libraries which might take a while), first create a new virtualenv and activate it.

Navigate to project root level 
$ cd culture-amp-coding-test

Install the requirements inside the created virtualenv
$ pip install -r requirements.txt

Test the project which should pass all tests before running the project
$ python test_cliy.py

Create a file (txt) and add the test questions per line at input_file_path 
Run the cli application
$ python cli.py input_file_path

The cli accepts an additional parameter n = top number of questions matched, which by default is 2 
$ python cli.py input_file_path -n 3

The output is in JSON format which by default is to stdout. It can be redirected to an output file using 
$  python cli.py input_file_path > output_file_path
