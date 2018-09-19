
# Measuring Similarity with Euclidean Distance
## Objectives

* Understand the importance of using similarity measures in data analysis. 
* Describe similarity between data objects based on the values of their contained features.
* Measure the distance between two data objects in euclidean space using Pythagoras Theorem.
* Understand the wider applications of using Euclidean distance based similarity. 




## Introduction 

This lesson provides an introduction to the concept of *similarity* or *distance* measures which is used heavily in data science domain for a number of applications. The lesson will give you an intuitive description for calculating Euclidean distance as a similarity measure, as well as it's implementation in python.

#### Similarity is the measure to identify how alike two data objects are. Dis-similarity, on the other hand could be thought of as a measure of how two data objects differ.

Similarity measures are used in a number of supervised/un-supervised machine learning tasks e.g. unsupervised clustering and classification algorithms are based on calculating the similarity between given data points. If our dataset is numeric and can be shown in an n-dimensional space, then there are various geometric metrics we can use to collectively process the data elements. 

> An n-dimensional space can be thought of as a multidimensional scatter plot that can be used to plot n number of dimensions of data (as compared to two/three dimensional scatter plots that we normally see).

A 3D scatter plot may look like the example below showing data dimensions with respect to mileage, power and fuel consumption. We can not view data beyond 3 dimensions, however we can process such data numerically as we shall see. 

![](https://d2mvzyuse3lwjc.cloudfront.net/doc/en/Tutorial/images/3D_Scatter_with_Colormap/3D_Scatter_with_Colormap.png)

### Similarity measures in data mining

Data mining often deals with identifying relationships between data objects. Similarity measures in the data mining domain are used to represent a distance, with data dimensions as features of the objects. Small and large values of these distance measures are seen as an indication of high and low similarity respectively. Similarity is a relative measure and depends highly on the domain and application. For example, two cars can be similar because of their color, engine size or price. Hence we need to be very careful when calculating distance across features that are unrelated. Normalized Similarity is usually measured as a value between 0 and 1 (inclusive), where 1 represents a high similarity (X=Y) and 0 - no similarity at all (X$\neq$Y).  

Let's Look at the most popular similarity distance measure which are commonly used in machine learning systems. 

## Euclidean Distance 

When measuring the distance between data objects in the plane, we we tend to use the Euclidean distance.It is the most commonly-used of our distance measures and hence often just referred to as “distance”. When data is dense or continuous, this is the best proximity measure. The Euclidean distance between two points is the length of the path connecting them. This measure is based mainly on the Pythagorean theorem as shown in the diagram below:

![](ed1.png)


According to this approach, The distance between two points (x1,y1) and (x2,y2), on a two dimensional plane is treated as a hypotenuse of a right angled triangle. This is broken down into its components of base and perpendicular. The distance $d$ between these points, following Pythagoras theorem can be simply calculated as: 

### $$d((x_1,y_1),(x_2,y_2)) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$

A general notation for this distance `d` for `n dimensional space` between two points `p` and `q`,  can be shown as:

$$d = \sqrt{\sum_{i=1}^{n}(q_i-p_i)^2}$$

So in simplest form, this measure shows an ordinary straight-line distance between two points in Euclidean space. This is all it takes to calculate the most commonly used distance measure. How about implementing this measure using python and numpy.


```python
import numpy as np
import math

def distance(p,q):
    d = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p, q)]))
    return d

# Calculate Eucledian distance
a = [1,2,3,4,5]
b = [6,7,8,9,10]
c = [4,5,6,7,8]

distance(a,b), distance(a,c), distance (b,c)
```




    (11.180339887498949, 6.708203932499369, 4.47213595499958)



### Verify Results

So we see with this simple example, that vectors that are alike give a high value, whereas vectors with different feature values output a low value for this distance measure. This functionality is available in python in a number of modules. Below we shall use `scipy.spatial.eucledian(x,y)` to verify our answer.


```python
from scipy.spatial import distance
distance.euclidean(a, b), distance.euclidean(a, c), distance.euclidean(b, c)
```




    (11.180339887498949, 6.708203932499369, 4.47213595499958)



Spot on. So how you've learned this handy technique to identify whether two data points are similar or different in terms of their features. We shall use this technique to perform a basic similarity based grouping/clustering task in the following lab. 


## Summary

In this lesson we saw how to use Euclidean Distance as a similarity measure to identify how alike data objects are. In the following lab, we shall use this measure for a simple problem and see how we can analytically identify closeness of real world data points in a euclidean space.
