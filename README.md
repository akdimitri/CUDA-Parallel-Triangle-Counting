# CUDA-Parallel-Triangle-Counting

author: Antoniadis Dimitrios

e-mail: akdimitri@auth.gr

University: Aristotle University of Thessaloniki (AUTH)

Subject: Parallel and Distributed Systems

Semester: 8th

---

**Description**: this repository includes the source code and other files of the fourth (4th) Assignment. Aim of the assigment is to implement a program that counts the number of the triangles of a simple undirected graph G(V,E) using CUDA programming language. This repository includes three (3) implementations in CUDA fulfiling the above objective. Furthermore, a sequential algorithm is also included in this repository.

To test the efficiency of the implemented programs the following Graphs were examined:

  * [auto](https://sparse.tamu.edu/DIMACS10/auto)
  * [great-britain_osm](https://sparse.tamu.edu/DIMACS10/great-britain_osm)
  * [delaunay_n22](https://sparse.tamu.edu/DIMACS10/delaunay_n22)
  * [delaunay_n23](https://sparse.tamu.edu/DIMACS10/delaunay_n23)
  * [fe_tooth](https://sparse.tamu.edu/DIMACS10/fe_tooth)
  * [144](https://sparse.tamu.edu/DIMACS10/144)
  * [citationCiteseer](https://sparse.tamu.edu/DIMACS10/citationCiteseer)
  * [road_central](https://sparse.tamu.edu/DIMACS10/road_central)
  * [germany_osm](https://sparse.tamu.edu/DIMACS10/germany_osm)
  * [road_usa](https://sparse.tamu.edu/DIMACS10/road_usa)
  
The efficiency of the algorithms(execution time) was compared to the MATLAB execution time of the following algorithm.
![Triangle Counting Algorithm](https://github.com/akdimitri/CUDA-Parallel-Triangle-Counting/blob/master/images/algorithm.png)

---

**Results**

The following graph shows the comparison of the 4 implemented algorithms and the algorithm executed by MATLAB for graph [auto](https://sparse.tamu.edu/DIMACS10/auto). On y axis is time in seconds and on x axis is the name of the algorithm.

![Algorithms Comparison](https://github.com/akdimitri/CUDA-Parallel-Triangle-Counting/blob/master/images/auto.png)

The following graphs show the speed up [_CUDA_](https://github.com/akdimitri/CUDA-Parallel-Triangle-Counting/blob/master/code/CUDA%20parallel/main.cu) algorithm achieved over _MATLAB_. On the y axis is the speed up (times faster), on the x axis is the name of the graph.

![CUDA/MATLAB Comparison speed up](https://github.com/akdimitri/CUDA-Parallel-Triangle-Counting/blob/master/images/speed_up.png)

![CUDA/MATLAB Comparison speed up](https://github.com/akdimitri/CUDA-Parallel-Triangle-Counting/blob/master/images/speed_up_2.png)



