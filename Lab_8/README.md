# Lab 8: Data Analysis Examples

For this lab, I followed the instructions for installing TensorFlow on Debian Bullseye and ran the example code on my Raspberry Pi

## Numpy Array

```
bird@humming:~ $ python3
Python 3.9.2 (default, Mar 12 2021, 04:06:34) 
[GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> a = np.range(6)
>>> a = np.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])

```
```
>>> b = np.arange(12).reshape(4, 3)
>>> b
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
  
```
```
>>> c = np.arange(24).reshape(2, 3, 4)
>>> c
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
```
```
>>> b.shape
(4, 3)
>>> b.reshape(-1)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> b.rehsape(-1,1)
>>> b.reshape(-1,1)
array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10],
       [11]])
>>> b.reshape(2, -1)
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
       
```
```
>>> d = np.array([20, 30, 40, 50])
>>> d
array([20, 30, 40, 50])
>>> e = np.arange(4)
>>> e
array([0, 1, 2, 3])
>>> f = d-e
>>> f
array([20, 29, 38, 47])
>>> e**2
array([0, 1, 4, 9], dtype=int32)
```
```
>>> A = np.array([[1, 1], [0,1]])
>>> B = np.array([[2,0], [3,4]])
>>> A
array([[1, 1],
       [0, 1]])
>>> B
array([[2, 0],
       [3, 4]])
>>> A*B
array([[2, 0],
       [0, 4]])
>>> A.dot(B)
array([[5, 4],
       [3, 4]])
>>> np.dot(A,B)
array([[5, 4],
       [3, 4]])
```
```
>>> g = np.ones((2,3), dtype=int)
>>> g
array([[1, 1, 1],
       [1, 1, 1]])
>>> h = np.random.random((2, 3))
>>> h
array([[0.70808437, 0.08466636, 0.28659055],
       [0.5200572 , 0.74046818, 0.23931682]])
>>> g *= 3
>>> g
array([[3, 3, 3],
       [3, 3, 3]])
>>> h += g
>>> h
array([[3.70808437, 3.08466636, 3.28659055],
       [3.5200572 , 3.74046818, 3.23931682]])
```
```
>>> k = np.random.random((2,3))
>>> k
array([[0.62038486, 0.0447532 , 0.16935452],
       [0.67369966, 0.16229027, 0.66676298]])
>>> k.sum()
2.3372454889248333
>>> k.min()
0.044753203639362416
>>> k.max()
0.6736996629093664
```
```
>>> m = np.arange(12).reshape(3, 4)
>>> m
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> m.sum(axis = 0)
array([12, 15, 18, 21])
>>> m.min(axis = 1)
array([0, 4, 8])
>>> m.cumsum(axis = 1)
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]], dtype=int32)
```
```
>>> n = np.arange(5)
>>> n
array([0, 1, 2, 3, 4])
>>> n[[1, 3, 4]] = 0
>>> n
array([0, 0, 2, 0, 0])
```

## Screenshots

![Screenshot 2022-05-08 233442](https://user-images.githubusercontent.com/78375489/167952300-633f685d-73ab-4e38-b679-5e220bdb9ae2.jpg)

![Screenshot 2022-05-08 233617](https://user-images.githubusercontent.com/78375489/167952304-cf21eb0e-f1b6-4db6-a734-e53b01be9af2.jpg)

![Screenshot 2022-05-08 233706](https://user-images.githubusercontent.com/78375489/167952311-9e7bd57d-ad05-42c9-aa62-e76e8d0669f9.jpg)

![Screenshot 2022-05-08 233749](https://user-images.githubusercontent.com/78375489/167952316-975c352f-f249-431f-b8b3-0676beab0528.jpg)

![Screenshot 2022-05-08 233827](https://user-images.githubusercontent.com/78375489/167952322-17b6d9b1-88bb-460f-a550-742d783fa01d.jpg)

![Screenshot 2022-05-08 233858](https://user-images.githubusercontent.com/78375489/167952329-60660870-f60c-49ff-ad4a-775f8829fa78.jpg)

![Screenshot 2022-05-08 233927](https://user-images.githubusercontent.com/78375489/167952335-ebb9dd93-f37b-433b-a5e2-31b0cf455924.jpg)

![Screenshot 2022-05-08 233957](https://user-images.githubusercontent.com/78375489/167952339-8d8b380a-909c-49ea-b1ad-d38bcecdc298.jpg)

![Screenshot 2022-05-08 234047](https://user-images.githubusercontent.com/78375489/167952345-34781a54-0275-4f78-88ff-a0441b57003e.jpg)

![Screenshot 2022-05-08 234128](https://user-images.githubusercontent.com/78375489/167952350-b5e410c0-4ce9-4a1e-a41d-830acad2ca3a.jpg)

![Screenshot 2022-05-09 001907](https://user-images.githubusercontent.com/78375489/167952353-0b81b7c9-7a4f-4a28-bd6a-707d2b786638.jpg)

![Screenshot 2022-05-09 001954](https://user-images.githubusercontent.com/78375489/167952357-247cf5ea-e5a5-4a8d-aeed-64767471f993.jpg)

![Screenshot 2022-05-09 002029](https://user-images.githubusercontent.com/78375489/167952363-d02c631b-a278-4744-a49b-ccd1e2d05502.jpg)

![Screenshot 2022-05-09 002107](https://user-images.githubusercontent.com/78375489/167952369-7c802ea8-21c8-444a-9886-65119a3e9cbf.jpg)

![Screenshot 2022-05-09 002257](https://user-images.githubusercontent.com/78375489/167952371-1f72a1d6-8f5e-49a0-b928-30d7f7d898d7.jpg)

![Screenshot 2022-05-09 002612](https://user-images.githubusercontent.com/78375489/167952376-f0f213f5-a096-48c2-8e22-d496c42ccd7d.jpg)

![Screenshot 2022-05-09 002648](https://user-images.githubusercontent.com/78375489/167952379-70c00e19-484a-481f-9615-80d3d590f694.jpg)

![Screenshot 2022-05-09 002749](https://user-images.githubusercontent.com/78375489/167952387-341fa501-6fd2-4c7d-8d2b-8e95cd3e8f07.jpg)

![Screenshot 2022-05-09 002828](https://user-images.githubusercontent.com/78375489/167952393-4359c8fa-ca95-4c9d-9ab2-c5736af1045b.jpg)

![Screenshot 2022-05-09 002907](https://user-images.githubusercontent.com/78375489/167952395-c97f44d2-e069-45d9-a54e-9d6cbcb64e5b.jpg)

![Screenshot 2022-05-09 145023](https://user-images.githubusercontent.com/78375489/167952399-884956a0-95b1-4869-983e-19915c405449.jpg)

![Screenshot 2022-05-09 145112](https://user-images.githubusercontent.com/78375489/167952401-a202987d-0b5b-4c30-92fc-6eb87ab60e3d.jpg)

![Screenshot 2022-05-09 145151](https://user-images.githubusercontent.com/78375489/167952449-89ea1717-44b5-44ac-b433-8603b4f0378b.jpg)

![Screenshot 2022-05-09 151110](https://user-images.githubusercontent.com/78375489/167952457-0ddf49b5-a760-4a67-a35f-b252ea794817.jpg)

![titanic_1](https://user-images.githubusercontent.com/78375489/167952461-9b6c69cf-fcfc-4b50-8415-f0aee8bafa17.jpg)

![titanic_2](https://user-images.githubusercontent.com/78375489/167952465-e2181e75-7b33-45ea-96ea-8219d47a67a2.jpg)
