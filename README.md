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
```
n=4
clear = Cleanerrobot(MaxTrash=4*n)
clear.CreateMesh(n, number_trash=1)
clear.mesh

t1 = clear.AdMatrix
plt.imshow(clear.mesh, interpolation='nearest')
plt.savefig("test.png")

p= clear.TrashPoints[0]

print("Gerando as soluções")
a = clear.GenerateNSoluction(p, 100)

print("gerando as gerações")
x, best = clear.CreateNGenerations(n=100, return_fmean=True, return_fbest=True, population=2, elitismo=3)

df = pd.DataFrame(columns=["Generation", "Mean", "Best"])
df["Generation"] = np.arange(len(x))
df["Mean"] = x
df["Best"] = best
plt.plot(df["Generation"], df["Mean"], label="Média da geração")
plt.plot(df["Generation"], df["Best"], label="Melhor indivíduo")
plt.title("Avaliação das gerações")
plt.legend()
```

## TO DO
- [] Consider what otimizer we will use
- [] Implement the Branch-and-Bound method in directory functions
- [] Implement PlanoDCorte in directory functions
- [] Implement tests with result we lnow
- [] Implement non linear optimization
- 
