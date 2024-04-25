using LinearAlgebra

function GradeToA(M::Matrix)::Matrix
	#=
	Consideramos que recebemos uma malha com valores diferentes de 0 para os pontos que podemos passar
	assim, consideramos a matrix A de adjascência
	=#
	row, colum = size(M)
	A = zeros(row*colum, row*colum); #creation of matrix adjescense

	for i in 1:row
		for j in 1:colum
			if (M[i,j] != 0)
				M[ (row*i)+j, j] = 1
			if (j+1 <= colum & M[i,j+1] != 0)#Same row + 1colum 
				M[ (row*i)+j, j+1] = 1
			if j(-1 > 0 & M[i, j-1])#Same row, -1 colum
				M[ (row*i)+j, j-1] = 1
			if (j + colum < row*colum, & M[i-1,j]!=0)
				M[ (row*i)+j, j+colum] = 1
			if (j - colum < row*colum & M[i+1,j]!=0)
				M[ (row*i)+j, j-colum] = 1
		end
	end

	return A
end