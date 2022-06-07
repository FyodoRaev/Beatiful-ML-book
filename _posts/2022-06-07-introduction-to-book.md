---
layout: post
title: "Machine Learning Handbook"
author: "Fyodor Raevskiy"
categories: journal
tags: [documentation,sample]
---

<center>This is an online Machine Learning textbook made for people who are not frightened by mathematics and want to dive into ML technologies. You will learn classic theory and details of algorithms implementation going from the basics of ML and growing to the level where you are able to read sophisitaced scientific papers.</center>
<hr>

### Contents

## [Introduction](#introduction)

1. ## Classic Supervised Learning
   1. Linear Models
   2. [Metric algorithms]
   3. [Decision trees]
   4. [Ensembles]
   5. [Gradient boosting]
2. ## [Evaluating a model]
3. ## [Deep Supervised Learning]
   1.  [Introduction to a fully connected neural network]
   2.  [What is backpropagation really doing?]
   3.  [Details of Learning]
4. ## Probabilistic models
   1. [Introduction to Probabilistic models]
   2. [Exponential class of distributions]
   3. [Generalised Linear models]
   4. [How to estimate probabilites?]
   5. [Generative way of classification]
   6. [Bayesian estimation]
5. ## Practical chapters
   1. [Clusterization]
6. ## Theory behind the ML
   1. [Bias-variance decomposition]
7. ## Basic knowledge of Maths 
   1. [Matrix calculus]


<center>I will be updating this book and add new chapters so you should follow the news on this <a href="https://t.me/+Y93ppaidWEoyYWNi">telegramm channel</a> </center>


## Introduction
![image](https://ml-handbook.ru/chapters/intro/images/cover.png)
*Authors: Filipp Sinitsyn, Stanislav Fedotov*


# About this book
#### The book that you are reading now was created by a team of very kind people who graduated from the School of Data analysis (ШАД). It wouldn't appeared without two wonderful courses:
  1. [The course of Konstantin Vyacheslavovich Vorontsov who has brought up the majority of book's authors.](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%28%D0%BA%D1%83%D1%80%D1%81_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B9%2C_%D0%9A.%D0%92.%D0%92%D0%BE%D1%80%D0%BE%D0%BD%D1%86%D0%BE%D0%B2%29)
  2. [NLP Course For You](https://lena-voita.github.io/nlp_course.html) by Lena Voita, who is appreciated for showing us how should modern book looks like.

#### This book is just a typed version of Machine Learning course in SDA(School of Data Analysis, ШАД) which has simplified nothing. The goal is to give the knowledge, describe important algorithms and tell about practical methods, about real-world problems and implementations of algorithms.

#### Mathematics is just one of the languages which the book was written in. We will try to explain the topics properly, but you are still required to know linear algebra, calculus, and probability theory because it will make the reading experience easier. The knowledge of statistics and convex optimization isn’t necessary, although it will ease understanding of some concepts.


### So, let's begin.

<hr>

## __Machine Learning__
#### __Machine Learning__ is a science of algorithms that automatically improve through experience.

After the invention of computers, humanity tries to automate more and more of its needs. A lot of problems can be solved algorithmically, but not all of them. For example, there are problems that are even unsolvable for humans, moreover they are proved to be unsolvable efficiently and computer will not make a prodigy there (we talk about NP problems). However, there ARE problems that people don’t count as difficult ones but which are still tough to be coded:

 * translate text from one tongue to another

 * predict a disease with symptoms

 * range some documents in informational search

 * determine the object in a picture

 * estimate the flat’s price

Those problems can be combined with following features:
  
1. the solution can be represented as a function which maps __objects__, or samples, into __predictions__(targets). For instanse, diseased people into diagnosis, documents into relevancies  
2. we don’t need to have perfect solutions for them, just the good enough will be okay because even a real doctor sometimes makes mistakes.
3. we are supposed to have the right answers (e.g. label for the given image)
    
Function which maps objects into predictions is called __model__, the given bunch of samples with right answers is called __dataset__ or training set. The training dataset must include:

* objects (it could be images, flat’s features, heights or weight etc)

* answers for them (labels for images, flat’s price, diagnosis etc), answers are usually named as __targets__

We want to create a model using this training dataset which will be nice at predicting values. But what does being "nice" mean? People usually use different __metrics__ to evaluate the model's performance, i.e. functions that show if predicted values look like the right ones. 

There are a lot of different metrics:
* for the diagnosis problem metrics which return the proportion of the right-predicted diagnoses would suit really nice
* for the flat's prices problem - proportion of flats for which the difference between prediction and the right price is not above certain threshold
* for the documents ranging problem - proportion of documents that were sorted incorrectly
* 
Usually, goal is to get the best (the biggest or the lowest in different cases) value of metric. 

__Tricky question__.It is important to memorize that needs of your customers can be represented as different types of metrics. What metric would you choose in the following cases:
* Ordinary year in the usual therapeutic department of a regular hospital
* So disgusting illness that any people who are predicted to have it will be ashamed of themselves and get teased
* Predicting of a dangerous and highly contagious disease.
<details>
    <summary>Answer (don't open it immediately; firstly think for yourself!)</summary>
    Of course in different cases and problems we should use different metrics, these are examples of answers:
    <li>Ordinary hospital - then doctor will be satisfied if the proportion of right-predicted diagnoses is high (this metric is called accuracy</li>
    <li>Predicting this really unpleasent disease - then we should maximize the proportion of people who are predicted this illnes and indeed have it</li>
    <li>Finding people who have dangerous disease - then we must not miss a single defected. This metric can be also represented as  proportion of correctly identified media (this metric is called recall).</li>
    Obviously those are the most simple metrics and in real-life problems data scientists meet more sophisitaced hierarchy of metrics. We will discuss them more explicit in the chapter "Evaluating models".
</details>

Let's take the flat pricing problem as an example. Constant function - $f(x)=c$ - will be our model (i.e. we will predict the same price for every flat) mean absolute error will evaluate the model's performance. 

$MAE(f,X,y)=L(f,X,y)=\frac{1}{N}\sum_{i=1}^N \|f(x_i)-_i \| \rightarrow min_f,$

where $f$ - is the model, $X =\{x_1, ... , x_N \}$ - training dataset (data about flat that we managed to find), $y = \{y_1, ... , y_N \}$ - right answers (i.e. prices for those flats).  

Since the prediction of the model is constant, it is easy to take a derivative from it, which we equate to zero in order to find the optimal value of $c$:

$\nabla_cL(f,X,y)=\frac{1}{N}\sum_{i=1}^Nsign(c-y_i)=0$

Slightly pretending that we are fools at mathematical rigor and formality, we can say that 0 (and, accordingly, the optimum of our metric) is reached at the point $f(x)=median(y)$ 

__Tricky question__.Let's now consider the mean squared error (MSE) metric in the problem of predicting the price of flat:

$MSE(f,X,y)= \frac {1}{N} \sum_{i=1}^N(f(x_i)-y_i)^2$

What is the optimal value for parameter $c$ in this case?

It is a mean value: $c^*= \frac{1}{N}\sum_{i=1}^Ny_i$

Great, so we can find the optimal model among constant ones. Maybe it can be done in some more interesting class of models? This is the question that most of our book will be devoted to.
The classic ML course includes description of model classes and ways of working with them.

Despite the fact that to solve most practical problems today it is enough to know only two types of models - gradient boosting on decision trees and neural network models - we will try to talk about others in order to develop your deep understanding of the subject and give you the opportunity not only to use the best established practices, but also, if you wish, to participate in the development of new ideas and the search for new methods - already in the role of a researcher, not just an engineer.

Not every combination of problems, models and metrics makes sense.

For example, if you predict the hazard class of a substance (sometimes from 1st to 4th) according to its chemical formula, MAE by default does not seem to be a very reasonable metric, certainly less reasonable than the proportion of correct predictions: despite the fact that classes seem to be represented by numbers, their order does not have clear sense.

