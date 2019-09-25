# AI-Machine-Learning-Law
Summer Research 
This is a classifier which can read law documents of Veteran Disasbility appeals and will tell the user what type of rhetoric role each sentence had.


How to Use:

Run the Classify.py program then type load_file('sentiment.json')

This will load in a Json file which has been improved with sentiment analysis in order to increase the accuracy score of the classifier.

You can also choose the original JSON file by typing load_file('all_sentences.json') (not recommended since it gives lower accuracy scores)


After loading in your file you must choose which machine learning classifier you are going to use.

Type either go_log_reg() for logisitc regression and so on for the other functions that I added.


Lastly type try_fit() in order to run the model and get back an accuracy score, a confusion matrix, and the incorrectly labaled rhetoric roles of the sentences that way the user can look at them and try to see if they can see what would be causing the problems and causing them to be labaled incorrectly.


(you can also create the file sentiment.json by taking your raw JSON file 'all_sentences.json' and then run it rough sentiment.py which is a algorithm I wrote which will take the sentmient score of each sentence and then turn that number from -1 to 1 into a word such as POS A POS B POS C and the inverse so NEG A NEG B NEG C.  the letter is the magnitutde and goes up to 10 letters in both directinos depending how posotive or negagtive a sentence is.  This word is then tagged in the JSON file in order to add context and increase the accuracy of the algorithm.



The purpose of this study was to write an algorithm which could most accurately label different types of
sentences within legal documents. The highest accuracy scores were achieved by utilizing machine learning and artificial
intelligence. Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand,
interpret and manipulate human languages such as english in our study. Throughout our research several python
libraries for machine learning and NLP algorithms were needed including the Natural Language Toolkit (NLTK), Vader
Sentiment Analysis and Scikit-Learn. Scikit-Learn is an open source machine learning tool in python made to be simple
and efficient for data mining and data analysis. Using sentiment analysis we were able to improve our accuracy by over
four percent. Sentiment analysis is the process of computationally identifying and categorizing opinions expressed in a
piece of text. This lets our program reads a sentence and tell us if it was a positive, negative, or neutral statement. Once
we perform the sentiment analysis we receive back a number from the range negative one to a positive one showing how
negative or positive the statement was. After this we created a program which could take these numbers and then place
them on a scale from one through ten and depending where it was on a scale it received a different tag in the JSON file.
JSON is short for JavaScript Object Notation, and is a way to store information in an organized and easy to access
manner. It is tough for a computer to read a document like a human does that&#39;s why we reformat the original text into an
easy to read JSON file.
