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
