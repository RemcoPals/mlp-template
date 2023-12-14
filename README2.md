# Tweet Emoji Classification 

## The task
With this project we aim to produce a minimum viable product that serves as the baseline 
for a classification model. The exact task of the model is to read a Tweet and predict an emoji that fits it.

## Our model

**1. A short description of what you’ve already implemented. This helps your TA assess
whether things are on track.**

At first, we intended to use a linear regression model for the MVP, but after consideration, we decided to 
implement a variation of a Support Vector Machine. What we ended up using is a Support Vector Classifier (SVC),
with a non-linear Radial Basis Function kernel. (TODO remmco explain why).

The first step we took was to load the data and create a train-validation-test split.
We, then, converted the text data to numerical features using TF-IDF using the TdIdf vectorizer, which also acts as our preprocessing pipeline.
After the data was preprocessed, the SVC was trained and predictions were generated.

In order to assess the model performance, we generated a classification report which provides metrics such as accuracy, F1 score, precision and recall.

A confusion matrix was also created so that we could assess which features were predicted by the model and what exactly leads us to get the specific accuracy we got. (TODO probably re-write this)

**2. Evidence that you achieve above random guessing model performance on your
validation data. This can be your validation accuracy vs. random guess accuracy.**

Out data set contains 20 emojis (labels), therefore we are performing a multi-class classification.
Having 20 classes means that a model random guessing would produce an accuracy of 0.05, but since our 
most prominent feature makes up for 20% of the data, we will consider random guessing to be an accuracy of 20%.
That is because if the model was to only predict that label all the time, it would get an accuracy of 20%. 

As can be seen in our classification report, our testing accuracy is 24% which we consider better than random guessing.

**3. An outline of what you’re still planning to do. This helps your TA assess whether things
are on track.**

-word embedding
-generate more data to solve class imbalance
-other things

**4. A way for your TA to access your github repo (important!). This can be a link if your
repository is public (recommended), or you can add your TA’s github account as a
collaborator. Make sure this actually works. You can also arrange this with your TA
before the deadline!**

**5. A short statement describing what each team member has contributed.**
