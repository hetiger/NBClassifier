# NBClassifier
Naive Bayes Classifier for multi-class (truthful/deceptive and positive/negative) classification of hotel reviews.
<br/>
<br/>
•	Performed sentiment analysis of hotel reviews using word tokens as features for classification.<br/>
•	Performed add-one smoothing on training data and ignored unknown tokens in the test data.<br/>
<br/>
Core Technology: Python &amp; JSON.
<br/>
# Data
Deceptive Opinion Spam Corpus v1.4 available at http://myleott.com/op_spam/ 
<br/>
# Programs
1. *nblearn.py*<br/>
Learns a naive Bayes model from the training data and writes the model parameters to a json file called *nbmodel.txt*.<br/>
&gt; python nblearn.py /path/to/train/file /path/to/label/file.<br/><br/>
2. *nbclassify.py*<br/>
Reads the model parameters from the file *nbmodel.txt*, classifies each entry in the test data, and writes the results to a text file called *nboutput.txt* in the same format as the label file from the training data.<br/>
&gt; python nbclassify.py /path/to/test/file.<br/>
# Results
Results for classifier executed on unseen test data.

|           | Precision | Recall |  F1  |
|-----------|:---------:|:------:|:----:|
| Deceptive |    0.81   |  0.91  | 0.86 |
| Truthful  |    0.90   |  0.79  | 0.84 |
| Negative  |    0.95   |  0.92  | 0.94 |
| Positive  |    0.92   |  0.96  | 0.94 |

Weighted Avg.	**0.89**
