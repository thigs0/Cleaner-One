# Cleaner-One
Algorithm to find best path to cleaner your house using **Julia language**

The code above is based at inter optimization.
The place is considered a subset of points like image 
![Region](image/region.png)

without pass inside red objects and don't leave the domain.
Restricttions

- Storage trash limitation
- batery limitation
- Region Sleepest than others
- Exists times that are minus humans at place

### Implementation
To run, install 
``` using Pkg
    Pkg.add("LinearAlgebra")
    Pkg.add("Plots")
    Pkg.add("")
```
(Integer programing)[./functions/PI.jl]

    - Runs the algorith that resolve the matrix (Ax=b)
    - Return the best path

    
(Draw path)[./functions/DrawPath.jl]

    - Draw the path that the robot will follow and save as image

## TO DO
- [] Consider what otimizer we will use
- [] Implement the Branch-and-Bound method in directory functions
- [] Implement PlanoDCorte in directory functions
- [] Implement tests with result we lnow
- [] Implement non linear optimization
- 
