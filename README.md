# MachineLearningOctavecodes
Linear Regression, Logistic Regression, Neural Networks and Polynomial Regression







Abstract

The recent progress in machine learning makes it possible to design efficient algorithms for predictions and classification in many data sets. This project describes comparisons and methods to control the predictions developed by various algorithms. The focus here has been on supervised learning, with a brief introduction to unsupervised machine learning.
We have carried out comparisons  between popular techniques namely, linear vs polynomial regression (in single variable),  gradient descent vs normal equations, logistic regression vs neural networks(with and without back propagation).
Along side comparisons, we have also described the working of the aforementioned techniques.



Machine learning is used to build computer systems or machines: perception of their environment: vision, object recognition (faces, patterns, natural languages, writing, syntactic forms, etc.), search engines; diagnostic aids, including medical, bioinformatics, brain-machine interfaces, fraud detection in credit card, financial analysis, including analysis of the stock market, classification of the DNA sequences, set, software engineering, adaptive or better adapted websites, locomotion robots, and so on.


Machine Learning

Definition:
In 1959, Arthur Samuel stated that: machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.
We can re-interpret machine learning as the field of study interested in the development of computer algorithms for transforming data into intelligent actions. Machine learning lies at the intersection of computer science, engineering and statistics.
In 1998, computer scientist Tom M.Mitchell, said that a machine is said to learn if it is able to take experience and utilize it such that its performance improves up on similar experiences in future.
Description:
The basic learning process for a machine involves:-
·        Data input- it utilizes observation, memory storage, and recall to provide a factual basis for furthering reasoning.
·        Abstraction- it involves translating the data in broader representation, e.g. converting raw data into matrix form for efficient use by the machine learning algorithms.
·        Generalization-  using the abstracted data to form a basis for action.
 
During the process of knowledge representation, the computer summarizes raw inputs in a model, an explicit description of the structured patterns among data. There are many different types of models. Such as :
• Diagrams such as trees and graphs
• Equations
• Logical if/else rules
• Groupings of data known as clusters
 
Once a particular machine learning algorithm is chosen for a problem, we need to train the algorithm or allow it to learn. To train the algorithm, we need to feed it quality data known as training set. In a training set, the target variable is known. The machine learns by finding some relationship between the features and the target variable.
To test the machine learning algorithm, a separate data set is used, known as test set. Initially, the program is fed the training examples, for the machine learning to take place. Then, the test set is fed to the program such that the target value for the test set is not fed into the program. The target variable that the training example belongs to is then compared with the predicted value  and then, we can get an idea about the accuracy of the algorithm.
 

 




TYPES OF LEARNING















Supervised learning

In supervised learning, from a given set of data, the learning algorithm attempts to optimize a function (model) to find the combination of feature values that result in the target output.
There are many supervised machine learning tasks, such as,
·        Classification – this task involves predicting which category an example belongs to. The target feature to be predicted is known as class and is divided into categories known as levels. Algorithms that can be used to perform this task are:
o   Nearest neighbor
o   Naïve Bayes
o   Decision trees
o   Classification rule learner
·        Numeric prediction – this task involves forecasting of numeric data. Algorithms that can be used to perform this task are:
o   Linear regression
o   Multiple regression
o   Regression trees
o   Model trees
Apart from the algorithms mentioned above, there are many algorithms which can used to perform classification as well as numeric prediction. For instance, neural networks and support vector machines.





Unsupervised learning

The opposite of supervised learning is unsupervised learning. In this kind of learning, there is no label or target value for the data. There are many supervised machine learning tasks, such as,
·        Clustering – Here, the task is to group similar items together. K-Means algorithm is used for clustering purpose.
·        Pattern detection – the task involves identifying frequent associations within the data. The algorithm used for this purpose can be Association rules.
 
Machine learning has been used widely to:
·        Foresee criminal activity.
·        Examine customer churn.
·        Identify and filter spam messages from emails.
·        Create auto-piloting planes and auto-driving cars.
·        Target advertising to specific types of customers.
·        Automate traffic signals according to the road conditions.
·        Predict the outcome of elections.
 








Linear Regression & Polynomial Regression



















The concept of regression is concerned with determining the relationship between the dependent entity (the value to be predicted) and one or more independent entities.
Regression analysis commonly finds its use in modeling complex relationships among data elements, estimating the impact of a treatment on an outcome, and extrapolating into the future.
Regression can be of two types: linear regression and multiple regression.
Linear regression is used when the data set has only one feature (independent variable). On the other hand, multiple regression is used when the data set has more than one feature.
Regression equations need not be linear, that is, they can be of higher degree as well. Such models come under multiple regression.
A special type of multiple regression is polynomial regression with only one independent variable. Here, the higher powers of the independent variable behave as different variables in determining the output.
Linear Regression (single variable)
In this case, our training set is one dependent (say, y) as well as independent variable (say, x), that is we one value of y for a value of x.
Let us assume a function: hƟ(x), defined as :
                                            
	hƟ(x) = Ɵ0 + Ɵ1x  
hƟ(x) is known as hypothesis, and Ɵ0  and Ɵ1 are called parameters.
The idea here, is to choose and Ɵ0  and Ɵ1  so that hƟ(x)  is close to y for our training example (x,y).
In order to determine the optimal estimates of Ɵ0  and Ɵ1  , an estimation method known as ordinary least squares (OLS) is applied. We, estimate the square of the difference between the actual value of y and the predicted value, i.e., hƟ(x) and y, for each instance (x,y).
Using this , we define a cost function, J (Ɵ0 ,Ɵ1), as:

J (Ɵ0 ,Ɵ1) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2  

Hence, our goal is to minimize J (Ɵ0 ,Ɵ1) over Ɵ0  and Ɵ1  .

Polynomial Regression(single variable)
Similar to what we just discussed earlier, we shall have a hypothesis hƟ(x) defined as:
                    	hƟ(x) = Ɵ0 + Ɵ1x + Ɵ2x2 + Ɵ3x3 + … + Ɵnxn
and the parameters as: Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3, … Ɵn
the cost function will be:
                                	J (Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3 ,… Ɵn) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2









Gradient Descent vs Normal Equation


















We still have to find a way to minimize the cost function, whether it is linear regression or polynomial regression.
One such way is Gradient Descent.
What gradient descent says is :
·        Start with a Ɵ0 ,Ɵ1.
·        Keep changing Ɵ0 ,Ɵ1  to reduce J (Ɵ0 ,Ɵ1) until hopefully ending up at a minimum.
 
 
The above points can be mathematically interpreted as,
Repeat until convergence {
        	Ɵj  := Ɵj  - α ((∂/∂Ɵj) J (Ɵ0 ,Ɵ1))                                     	(simultaneously update
                                                                                                        	j = 0 and j = 1)
}
 
The point to remember is that:
·        If α is too small, gradient descent can be slow.
·        If α is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.
For linear regression, gradient descent can formulated as:
 
Repeat until convergence {
        	Ɵ0  := Ɵ0  - α (1/m)∑i=1m(hƟ(x(i)) – y(i))                        	(simultaneously update
        	Ɵ1  := Ɵ1  - α (1/m)∑i=1m(hƟ(x(i)) – y(i)).x(i)      	                                	j = 0 and j = 1)
}
 
For polynomial regression,
Cost function: J (Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3 ,… Ɵn) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2
Parameters: Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3 ,… Ɵn à this can be considered as an nth dimensional vector Ɵ
Hence we have,
        	J (Ɵ) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2
 
Gradient descent is formulated as:
 
Repeat until convergence {
        	Ɵj  := Ɵj  - α ((∂/∂Ɵj) J (Ɵ))                                 	(simultaneously update
                                                                                                        	j = 0,1,……,n)
}
 
This becomes,
Repeat until convergence {
        	Ɵj  := Ɵj  - α (1/m)∑i=1m(hƟ(x(i)) – y(i)).(x(i)) j                           	(simultaneously update
        	                                                        	                                	j = 0, 1…, n)
}
There is yet another way to solve for Ɵ, in order to get the minimum cost function, known as Normal Equation.
Normal equation is a method to solve  for Ɵ analytically.
Lets say,
        	J(Ɵ) = aƟ2 + bƟ + c
Set,
        	(d/dƟ)J(Ɵ) = 0
And solve for Ɵ.
 
Similarly, for
                    	J (Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3 ,… Ɵn) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2
Set,
        	(d/dƟj)J(Ɵ) = 0                            	(for every j)
 
Solve Ɵ0 , Ɵ1, Ɵ2 ,Ɵ3 ,… Ɵn
Instead of solving so many differential equations, the following equation may help:
        	Ɵ = (XTX)-1XTy
This is known as the normal equation.
 
Here, X is the data matrix of the dimension m X n.(m is the number of instances and n is the number of features)
 
Gradient Descent
Normal equation
·        Need to choose a proper value of parameter ‘α’
·        Needs many iterations everytime
·        Works well even when n is large
·   Needs more number of iterations(more time) for accuracy
·        No need to choose any paramter like ‘α’
·        Don’t need to iterate every time
·        Slow if n is large
·        Needs more memory to compute (XTX)-1
 
 







Data Set Information:

The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.
A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another. While the Vacuum is colected from and has effect on the Steam Turbine, he other three of the ambient variables effect the GT performance.
For comparability with our baseline studies, and to allow 5x2 fold statistical tests be carried out, we provide the data shuffled five times. For each shuffling 2-fold CV is carried out and the resulting 10 measurements are used for statistical testing.
We provide the data both in .ods and in .xlsx formats.(1)

Attribute Information:

Features consist of hourly average ambient variables 
- Temperature (T) in the range 1.81°C and 37.11°C,
- Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
- Relative Humidity (RH) in the range 25.56% to 100.16%
- Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
- Net hourly electrical energy output (EP) 420.26-495.76 MW
The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization.(1)
OUTPUT:

Gradient Descent



Normal Equation




Linear Regression vs Polynomial Regression (Single Variable)









Data Set Information:
The data set contains 12 data points. It stores the information about the amount of water flowing out of a dam using the change of water level in a reservoir.
The dataset contains historical records on the change in the water level, x, and the amount of water flowing out of the dam, y.
This dataset is divided into three parts:
A training set that your model will learn on: X, y
A cross validation set for determining the regularization parameter: Xval, yval
A test set for evaluating performance. These are unseen examples which our model did not see during training: Xtest, ytest(2)











Linear Regression with varying parameters

















Data Set Information:

The NASA data set comprises different size NACA 0012 airfoils at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.

Attribute Information:

This problem has the following inputs: 
1. Frequency, in Hertzs. 
2. Angle of attack, in degrees. 
3. Chord length, in meters. 
4. Free-stream velocity, in meters per second. 
5. Suction side displacement thickness, in meters. 

The only output is: 
6. Scaled sound pressure level, in decibels.(6)














Output:
No feature scaling








 

With Feature scaling






 


Linear Regression:
For different Regularization parameters















Having too many features, the learned hypothesis may fit the training set very well but will fail to generalize to new examples. This is known as overfitting.

Underfitting



Optimal Fitting

Over fitting


To overcome the problem of overfitting, we can do either of the following:-
Reduce the number of features by manually selecting which features to keep.
Regularization, i.e.  keeping all the features but reducing the magnitude of the parameters Ɵj .

By choosing small values for Ɵj , the hypothesis becomes simples and hence less prone to overfitting.

Lets take the case of linear regression.

The cost function of simple linear regression is:
	J (Ɵ0 ,Ɵ1) = (1/2m)∑i=1m(hƟ(x(i)) – y(i))2

We modify this function a bit, and the cost function for regularised linear regression becomes,

	J (Ɵ0 ,Ɵ1) = (1/2m)[ ∑i=1m(hƟ(x(i)) – y(i))2 + λ∑i=1nƟj2 ]


where,  λ∑i=1nƟj2  is known as regularization parameter.

It is extremely important to note here, that very large value of λ will result in underfitting because the cost function will approximately be of type 

y ≈ λ

resulting in a plot of something like this,



Data Set Information:
The data set contains 12 data points. It stores the information about the amount of water flowing out of a dam using the change of water level in a reservoir.
The dataset contains historical records on the change in the water level, x, and the amount of water flowing out of the dam, y.
This dataset is divided into three parts:
1.	A training set that your model will learn on: X, y
2.	A cross validation set for determining the regularization parameter: Xval, yval
3.	A test set for evaluating performance. These are unseen examples which our model did not see during training: Xtest, ytest(2)






OUtPUTS are given below:
Lambda = 0




Lambda = 1




























Logistic Regression:
For different Regularization parameters




Logistic regression analysis is concerned with determining the class to which a given test instance belongs.

In this kind of regression analysis, the training set consists of instances with a number of features, one of which is the class into which they fall. 
However, the set contains all the features except for the class, pretty obivous.

The job of logistic regression is to predict the class for the every instance in the test set.

Considering the case where the number of classes is two, y = {0,1}
The threshold classifier output hƟ(x) at 0.5:
If hƟ(x) ≥ 0.5, predict “y=1”
If hƟ(x) < 0.5, predict “y=0”

Under the logistic regression model,
	0 ≤ hƟ(x) ≤ 1

hƟ(x) = g(ƟTx)
g(z) = 1/(1+e-z)
 
Here, g is known as the sigmoid function and hƟ(x)  denotes the estimated probability that y=1, given the feature vector ‘x’ parameterized by the vector ‘Ɵ’.

Now the cost function J (Ɵ) is determined as:
	J (Ɵ) = (1/m)∑i=1mCost(hƟ(x(i)), y(i))

Cost(hƟ(x), y) = -log(hƟ(x))			if y=1
		    = -log(1- hƟ(x))		if y=0

Simplification this will result into,

Cost(hƟ(x), y) = -[ ylog(hƟ(x)) + (1-y)log(1- hƟ(x)) ]

Therefore, 
	J (Ɵ) = (1/m)∑i=1mCost(hƟ(x(i)), y(i))
J (Ɵ) = -(1/m)∑i=1m[ y(i)log(hƟ(x(i))) + (1-y(i))log(1- hƟ(x(i))) ]	

To fit the parameters Ɵ, we need to find the value of Ɵ vector for which 
J(Ɵ) has the minimum value.

To find minƟ J(Ɵ), we can either use gradient descent method or normal equation method.

After getting Ɵ, we can make a prediction of a new x, by finding the value of hƟ(x) as,

hƟ(x) = 1/(1+ exp(-ƟTx)) 












Problem 1

Data set Information:
This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other two. 

Predicted attribute: class of iris plant. 

This is an exceedingly simple domain. 

This data differs from the data presented in Fishers article (identified by Steve Chadwick, spchadwick '@' espeedaz.net ).

Attribute Information:

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica(4)

Iris Setosa has been assigned class 0. Iris Versicolou has been assigned class 1. 
Iris Virginica has been assigned class 2.





Output:

Lambda = 0



Lambda = 0.5



Lambda = 1.0





Lambda = 2.0




Problem 2

Data Set Information:

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

Class Distribution: number of instances per class

      	class 1 59
	class 2 71
	class 3 48

The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it ) 
1) Alcohol 
2) Malic acid 
3) Ash 
4) Alcalinity of ash 
5) Magnesium 
6) Total phenols 
7) Flavanoids 
8) Nonflavanoid phenols 
9) Proanthocyanins 
10)Color intensity 
11)Hue 
12)OD280/OD315 of diluted wines 
13)Proline 

In a classification context, this is a well posed problem with "well behaved" class structures. A good data set for first testing of a new classifier, but not very challenging.(5)

Output:

Lambda = 0



Lambda = 0.5







Lambda = 1.0



Lambda = 1.5



Lambda = 2.0





Neural Networks

Algorithms that come under the concept of neural networks are those algorithms that try to mimic the brain.

Such algorithms generally produce non-linear hypothesis.
Similar to the brain, that uses a network cells called neurons that are interconnected and together work as a massive parallel processor, the artificial neural network uses  a network of artificial neurons or nodes to solve the learning problems.
Considering the structure of a neuron, an input signal is received by the dendrites of the cell through a biochemical process that allows the signal to be weighted according to its relative importance or frequency.

Soon due to the accumulation of the input signals, a threshold is reached and then the neuron transmits an output signal down the axon. On reaching the axon terminals, this electrical signal is sent to another neuron via a tiny gap called synapses.

Similar to the signal processing described above, an artificial neuron/node has its dendrites’ signal weighted according to its importance.
Then the input signals are summed by the cell and the signal is transmitted on to another node according to an activation function.



Here,
Xi ‘s are input signals.
wi ‘s are the weights assigned to each input signal.	
f is the activation function.
y is the output signal.

And,
	y(x) = f(∑i=1n wi xi )

The artificial Neural Network has the following basic characteristics:-
Activation Function -> this function maps the set of input signals into a singal output signal.
Network Architecture -> this defines the number of neurons , number of layers and the way they are interconnected.
Training algorithm -> this defines how the connection weights are set.



Here, 
Layer 1 is known as the input layer.
Layer 2 and Layer 3 are called hidden layers.
Layer 4 is known as the output layer.


The activation function( hypothesis ) can be chose on their ability to describe the mathematical characteristics and relationships among data. Examples include unit step function, sigmoid function etc.



The cost function of a neural network is given as,




where,


and ‘j’ denotes the layer number, ‘i’ denotes the neuron.

Here also, to find minƟ J(Ɵ), we can use gradient decent method.








Logistic Regression vs Neural Networks















Data set Information:

Data set is in ex3data1.mat(3) and contains 5000 training ex-
amples of handwritten digits, which is a subset of the MNIST handwritten digit dataset. The .mat format means that that the data has been saved in a native Octave/Matlab matrix format, instead of a text (ASCII) format like a csv file.

There are 5000 training examples in ex3data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is \unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a 0 digit is labeled as 10, while the digits 1 to 9 are labeled as 1 to 9 in their natural order.



OUTPUT:

LOGISTIC REGRESSION- ONE VS ALL (Multiclass Classification)










NEURAL NETWORK





Neural Networks:
Changes when we apply Backpropagation

The backpropagation algorithm was originally introduced in the 1970s, but its importance wasn't fully appreciated until a famous 1986 paper by David Rumelhart, Geoffrey Hinton, and Ronald Williams. That paper describes several neural networks where backpropagation works far faster than earlier approaches to learning, making it possible to use neural nets to solve problems which had previously been insoluble. Today, the backpropagation algorithm is the workhorse of learning in neural networks.

The reason, of course, is understanding. At the heart of backpropagation is an expression for the partial derivative ∂J/∂wof the cost function J with respect to any weight w (or bias b) in the network. The expression tells us how quickly the cost changes when we change the weights and biases. And while the expression is somewhat complex, it also has a beauty to it, with each element having a natural, intuitive interpretation. And so backpropagation isn't just a fast algorithm for learning. It actually gives us detailed insights into how changing the weights and biases changes the overall behaviour of the network. That's well worth studying in detail.
With that said, if you want to skim the chapter, or jump straight to the next chapter, that's fine. I've written the rest of the book to be accessible even if you treat backpropagation as a black box.(7) 

Let's begin with a notation which lets us refer to weights in the network in an unambiguous way. We'll use wljk to denote the weight for the connection from the kth neuron in the (l−1)th layer to the jth neuron in the lth layer. So, for example, the diagram below shows the weight on a connection from the fourth neuron in the second layer to the second neuron in the third layer of a network:




One quirk of the notation is the ordering of the j and k indices. You might think that it makes more sense to use j to refer to the input neuron, and k to the output neuron, not vice versa, as is actually done. I'll explain the reason for this quirk below.
We use a similar notation for the network's biases and activations. Explicitly, we use blj for the bias of the jth neuron in the lth layer. And we use alj for the activation of the jth neuron in the lth layer. The following diagram shows examples of these notations in use:









Data set Information:

Data set is in ex3data1.mat(3) and contains 5000 training ex-
amples of handwritten digits, which is a subset of the MNIST handwritten digit dataset. The .mat format means that that the data has been saved in a native Octave/Matlab matrix format, instead of a text (ASCII) format like a csv file.

There are 5000 training examples in ex3data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is \unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a 0 digit is labeled as 10, while the digits 1 to 9 are labeled as 1 to 9 in their natural order.














Output:

 









Grouping together unlabeled items using k-means clustering algorithm

Clustering is a type of unsupervised learning that automatically forms clusters of “similar” things. It can be assumed as automatic classifier.

Here, we will be using k-means algorithm to accomplish the task of clustering. This algorithm is called so because it finds ‘k’ unique clusters and the representative of each cluster is a central value (mean) of the values in that cluster.

The number of clusters ‘k’ are defined by the user. The representative of each cluster is known as its Centroid. 

The following is the basic procedure of k-means:-
·         Initially, create k centroid points randomly.
·         Each point is assigned to a cluster. This is done by finding the closest centroid and assigning the point to that cluster.
·         The proximity of a point to a cluster is measured by calculating the Euclidean distance between them.
·         When this assignment is done, all the centroids are updated by taking the mean value of all the points in their respective clusters.
·         The procedure is repeated until there is no change in cluster assignment for any point.





Data set Information:
Data is in “Wholesale customers data.csv”  and contains
440 examples, each with 8 features and has no missing values. Each feature is of type integer.
Feature Information:
1)	FRESH: annual spending (m.u.) on fresh products (Continuous); 
2)	MILK: annual spending (m.u.) on milk products (Continuous); 
3)	GROCERY: annual spending (m.u.)on grocery products (Continuous); 
4)	FROZEN: annual spending (m.u.)on frozen products (Continuous) 
5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous) 
6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous); 
7)	CHANNEL: customersâ€™ Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal) 
8)	REGION: customers Region Lisnon, Oporto or Other (Nominal) 
Descriptive Statistics: 

(Minimum, Maximum, Mean, Std. Deviation) 
FRESH (	3, 112151, 12000.30, 12647.329) 
MILK	(55, 73498, 5796.27, 7380.377) 
GROCERY	(3, 92780, 7951.28, 9503.163) 
FROZEN	(25, 60869, 3071.93, 4854.673) 
DETERGENTS_PAPER (3, 40827, 2881.49, 4767.854) 
DELICATESSEN (3, 47943, 1524.87, 2820.106) 

REGION	Frequency 
Lisbon	77 
Oporto	47 
Other Region	316 
Total	440 

CHANNEL	Frequency 
Horeca	298 
Retail	142 
Total	440 


OUTPUT
On x-axis we have instance index and on y-axis , we have cluster index.

Clustering index begins from 0 and ends at k-1, for k number of clusters.

Three Clusters (k=3)

Five Clusters(k = 5)










Applications of Machine Learning

1) Automatic speech recognition -- speech to text

2) Automatic Voice/Face/Fingerprint recognition -- authenticating the user of a device--laptop/mobile/doors, etc.

3) Natural Language Processing -- programmatic sentiment analysis, intent analysis, statistical machine translation -- (google translate), etc

4) Automatic Medical diagnostics --detecting diseases from symptoms..

5) Automatic packaging plants --  example: packing different fishes on a conveyer belt into different cans automatically -- (from the book duda hart and stock)

6) Email spam detection. -- google, yahoo, etc use it.

7) Advertisements (Google Adsense, sponsored ads)/Recommendations 
engines (netflix)

8) Bioinformatics/computational biology

9) Computational Neuroscience -- (my ME major project dealt with one such application where it was required to "data mine" the neuronal connectivity from the neuron spike trains data)

10) Content (image, video, text) categorization

11) Suspicious activity detection from CCTVs -- witnessed this research being carried out by a friend in my alma mater using ML/ probability models
12) Frequent pattern mining -- market basket analysis -- keeping frequently bought items close together in a supermarket --to a) increase the purchases per unit time spent and b) to recommend that "you might want this item too" . Frequent pattern mining is not computationally trivial as it may sound like to the layman.

13) Largely --  all applications of the standard Machine learning/Data mining techniques -- clustering, classification, regression, frequent episode, frequent pattern, probability models, etc

14) All of the following functions common to most of our smartphones today utilize it.

•    For example while using our mobiles to type text, suggestions of words are produced by algorithms mainly using Markov chains with a graph type N-Gram.
•    The handwriting recognition uses a pattern recognition algorithm and a Markov process for the recognition of natural language. Both algorithms combined contribute to greatly improve performance and efficiency.
•    Voice recognition, especially the isolation of the voice notes in a noisy environment, using a non-supervised learning algorithm consisting of neural networks (a model designed in the manner of a human brain).









Final Report

According to all the comparisons that are done among different Supervised Learning algorithms, we have reached the following conclusion:

In Linear Regression, Gradient Descent should be used when the number of features is high and the cost function can approach minima quickly. Whereas Normal equation should be used when the number of features and the training set is small.
Linear Regression should be used when it is possible to fit the data and the number of features is huge. Polynomial Regression should be used when the number of features is small and the training set is not over-fitted.
Logistic Regression is the most basic algorithm used for classification and when implemented over different layers provides us with Neural Networks.
Regularization parameter has a tremendous effect on both types of regressions(Linear as well as Logistic).
Neural Networks should be used when it is not possible to achieve the desired result with high accuracy. We can use Backpropagation algorithm to further improve the accuracy of the result.





Refrences

http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
https://spark-public.s3.amazonaws.com/ml/exercises/ml-006/mlclass-ex5-006.zip
https://spark-public.s3.amazonaws.com/ml/exercises/ml-006/mlclass-ex3-006.zip
http://archive.ics.uci.edu/ml/datasets/Iris
http://archive.ics.uci.edu/ml/datasets/Wine
http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
http://neuralnetworksanddeeplearning.com/chap2.html
https://archive.ics.uci.edu/ml/datasets/Wholesale+customers

Books referred:-
Machine Learning in Action by Peter Harrington, Manning Publications
Machine Learning in R by Brett Lantz.



