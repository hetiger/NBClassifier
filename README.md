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
Deceptive Opinion Spam Corpus v1.4 available at <a target="_blank" href="http://myleott.com/op_spam/">http://myleott.com/op_spam/</a> 
<br/>
# Programs
<br/>
<ol>
<li><i>nblearn.py</i></li>
  •	Learns a naive Bayes model from the training data and writes the model parameters to a json file called<i>nbmodel.txt</i>.<br/>
  &gt; python nblearn.py /path/to/train/file /path/to/label/file.<br/>
  <br/>
<li><i>nbclassify.py</i></li>
  •	Reads the model parameters the file<i>nbmodel.txt</i>, classifies each entry in the test data, and writes the results to a text file called<i>nboutput.txt</i> in the same format as the label file from the training data.<br/>
  &gt; python nbclassify.py /path/to/test/file
  <br/>
<br/>
