class Soluction():
  def __init__(self, mesh, caminhos=np.array([np.nan, np.nan], ndmin=2), vectors=np.array([np.nan, np.nan], ndmin=2)):
    self.f = 0 #Quantos pontos o robô andou nessa solução
    self.caminhos = caminhos
    self.vectors = vectors
    self.TrashMesh = mesh.copy()
    self.EndPoint = np.array([])



  def AddCaminho(self, caminho:np.ndarray, vectors:np.ndarray):
    """Adiciona um caminho na solução
    retorna 0 se não tem mais lixo na sala e 1 se tem
    """
    self.caminhos = np.append(self.caminhos, caminho, axis=0) #adiciona um caminho
    self.vectors = np.append(self.vectors, vectors, axis=0)
    self.caminhos = self.caminhos[1:]
    self.vectors = self.vectors[1:]

    self.EndPoint = caminho[-1]
    for i in np.arange(caminho.shape[0]):#Atualiza a mesh
      self.TrashMesh[ int(caminho[i][0]) ][ int(caminho[i][1]) ] = 0

    self.f += int(caminho.shape[0])
    return 0 if np.sum(self.TrashMesh) == 0 else 1

  def Mutation(self, Mutation_Chance=0.1): #Fazendo
    """Caso a probabilidade mande, reorganizamos os vetores do caminho n e atualizamos o caminho

    """
    for i in range( len(self.caminhos) ):
      if np.random.random() < Mutation_Chance: # Mutação acontece
        np.shuffle(self.vectors[i]) #Reorganiza o vetor

        #Atualiza o caminho
        new_path = np.zeros( self.caminhos.shape )
        new_path = self.caminhos[0] # O ponto inical é o mesmo
        pass
      pass

  def AnimTrash(self):#Fazendo
    fig = plt.figure()
    t = self.TrashMesh.copy()

    for i in range(len(self.caminhos)): #Seleciona um caminho
      for j in range( self.caminhos[i].shape[0]): #atualiza os pontos que o caminho passa
        t[int(self.caminhos[i][j][0]) ][int(self.caminhos[i][j][1]) ] = 0

      ani = FuncAnimation( fig, plt.imshow(t, interpolation='nearest'))
    plt.show()

  def IsValid(self, mesh):
      """verifica se o caminho percorrido é válido (Atende as restrições)
      retorna 0 se não tem mais lixo na sala e 1 se tem
      """
      copy = mesh.copy()
      for i in np.arange( self.caminhos.shape[0] ):
        copy[ int(self.caminhos[i][0]) ][ int(self.caminhos[i][1])] = 0

      return True if (np.sum(copy) == 0) and self.f > 0 else False


class Cleanerrobot():
  def __init__(self, MaxTrash:int):
    self.mesh = None # Mesh that wee will optimize
    self.AdMatrix = None #matrix of adjascence for the mesh
    self.TrashPoints = np.array([np.nan, np.nan], ndmin=2) #Matrix with points when we have trash
    self.soluctions = None #Armazenamos as soluções
    self.TrashMesh = None # Matrix with position when trash
    self.MaxTrash = MaxTrash # Max capacity to stock trash, or max number of point to full the self trash

  def __PointMeshToPointAdmatrix(self, point) -> int:
    """Get point of mesh and return the numeration of it
    """
    lin, colu = point
    return lin*self.mesh.shape[0]+colu

  def __TrashMesh(self) -> None:
    """When is pass the mesh, we construct the mesh where have trash. It's basically all point equal one
    """
    self.TrashMesh = self.mesh.copy()
    self.TrashMesh[self.TrashMesh != 1] = 0

  def __PointAdmatrixToPointMesh(self, point):
    """Get number of  position on Admatrix ad return the Point mesh position (x,y)
    """
    return (point // self.mesh.shape[0] -1 ,  point % self.mesh.shape[0])  ##Verificar

  def __GetTrashPoints(self) -> None:
    """
      Baseado na malha, percorre os extremos e coleta onde estão os pontos de lixo
    """
    lin, col = self.mesh.shape
    #Percorre as linhas
    for i in np.arange(lin):
      if self.mesh[i][0] == 2:
        self.TrashPoints = np.concatenate((self.TrashPoints, np.array([i, 0], ndmin=2) ), axis=0 )
      if self.mesh[i][col-1] == 2:
        self.TrashPoints = np.concatenate( (self.TrashPoints, np.array([i, col-1], ndmin=2)), axis=0 )
    #Percorre as colunas
    for i in np.arange(lin):
      if self.mesh[0][i] == 2:
        self.TrashPoints = np.concatenate( (self.TrashPoints, np.array([0, i], ndmin=2)), axis=0 )
      if self.mesh[lin-1][i] == 2:
        self.TrashPoints = np.concatenate( (self.TrashPoints, np.array([lin-1, i], ndmin=2)), axis=0 )

  def __f(self) -> int:
    fun=0
    for i in self.soluctions:
      fun += i.f
    return fun

  def BestSoluction(self, number=1):
    fun = np.inf
    best = np.array([])
    df = pd.DataFrame(columns=["fun", "posi"])

    for i,j in enumerate(self.soluctions):
      df.loc[-1] = [j.f, i]
      df.index += 1
    df = df.reset_index()

    df = df.drop(columns=["index"])
    df = df.sort_values(by='fun')
    df = df.reset_index()
    for i in np.arange(number):
      best = np.append(best, self.soluctions[ df.loc[i, "posi"] ])
    return best

  def __f_mean(self):
    return self.__f()/len(self.soluctions)

  def __GenerateTwoVectors(self):
    """Gera dois vetores que se anulam, exemplo, se move para cima e se move para baixo. Esquerda direita
    """
    if np.random.random() < 0.5:
      return (np.array([1, 0]), np.array([-1, 0]))
    else:
      return (np.array([0, 1]), np.array([0, -1]))

  def __AdMatrix(self) -> None:
    self.AdMatrix = np.zeros((self.mesh.size, self.mesh.size))
    lin, col = self.mesh.shape #tamanho da linha coluna
    for i in range(lin):
      for j in range(col):
        if i == 0 or j == 0 or i == lin-1 or j == col-1:
          pass
        else: #atualiza a matriz de adjascencia
          if self.mesh[i][j] == 1 and self.mesh[i+1][j] == 1:
            self.AdMatrix[i*lin+j][(i+1)*lin+j] = 1

          if self.mesh[i][j] == 1 and self.mesh[i-1][j] == 1:
            self.AdMatrix[i*lin+j][(i-1)*lin+j] = 1

          if self.mesh[i][j] == 1 and self.mesh[i][j+1] == 1:
            self.AdMatrix[(i)*lin+j][i*lin+j+1] = 1

          if self.mesh[i][j] == 1 and self.mesh[i][j-1] == 1:
            self.AdMatrix[i*lin+j][i*lin+j-1] = 1
      self.TrashPoints = self.TrashPoints[1:]

  def LoadMesh(self, mesh):
    """
      Carrega a malha que iremos otimizar as rotas
    """
    self.mesh = mesh
    self.__AdMatrix()
    self.__GetTrashPoints()
    self.__TrashMesh()

  def CreateMesh(self, n:int, number_trash = None):
    """Cria uma malha aleatória de tamanho n
    com
    """
    if number_trash is None:
      number_trash = n*2

    A = np.zeros((n,n))
    for i in np.arange(n):
      for j in np.arange(n):
        if j != 0 and i != 0 and j != n-1 and i != n-1:
          if np.random.random() < 0.90:
            A[i][j] = 1
        else:
          if np.random.random() < 0.3 and  number_trash > 0:
            A[i][j] = 2
            number_trash -= 1

    self.mesh = A
    self.__AdMatrix()
    self.__GetTrashPoints()
    self.__TrashMesh()
    if len(self.TrashPoints) ==0: #Caso não coloque nenhum ponto de lixo
      self.TrashPoints = np.array([0,0])
      self.mesh[0][0] = 2


  def GeneratePath(self, StartPoint=None, EndPoint=None, return_vectors=False):
    """
      Gera um caminho válido aleatório que segue as restrições dadas
      1) Precisa partir de um depósito de lixo e chegar em um depósito de lixo
      2) Não pode passar a capacidade de lixo do robo

      return_vector: Se irá retornar ou não os vetores de movimentação

      retorna um array de posições (x, y) na sequência do caminho
    """

    def FindPathCaseTwo(StartPoint:np.ndarray):
      """StartPoint == Endpoint
      """
      n = self.MaxTrash // 2 #quantidade máxima de caminhos que se anulam
      n = np.random.randint(1, n) # Pega aleatóriamente a quantidade de caminhos que vamos ter
      points = np.zeros((n*2+1,2))
      points[0] = StartPoint
      vectors = np.zeros((n*2, 2))
      for i in np.arange(n):
        vectors[2*i], vectors[2*i+1] = self.__GenerateTwoVectors()

      for _ in np.arange(n): #Embaralha os vetores
        np.random.shuffle(vectors)

      for i in np.arange(vectors.shape[0]):
        k = 1
        p = points[i] + vectors[i]
        if np.sum( p >= 0 ) >= 2 and np.sum( p < self.TrashMesh.shape[1] ) >=2: # O ponto é válido
          points[i+1] = p.copy() # Atualiza o ponto

        else: #O ponto i não é válido
          while (np.sum( p < 0 ) >= 1 or np.sum( p >= self.TrashMesh.shape[1] ) >=1) and (k+1+i < vectors.shape[0]): #Enquanto o ponto não é válido
            k+=1
            p = points[i].copy() + vectors[i+k].copy()

          points[i+1] = points[i].copy() + vectors[i+k].copy()
          vectors[i], vectors[i+k] = vectors[i+k].copy(), vectors[i].copy() #Troca os vetores

      if return_vectors == True:
        return (points, vectors)
      return points

    def FindPathCaseOne(StartPoint, EndPoint):
      point = EndPoint - StartPoint
      #Separa o ponto em vários vetores
      out = np.zeros(( int(np.absolute(p).sum()) , 2 ))
      t=0
      while abs(point[0]) > 0: #Enquanto podemos andar
        param = 1 if point[0] > 0 else -1
        out[t][0] = param
        point[0] -= param
        t+=1
      while abs(point[1]) > 0:
        param = 1 if point[1] > 0 else -1
        out[t][1] =  param
        point[1] -=  param
        t+=1
      #Embaralha a ordem dos vetores
      np.random.shuffle(out)
      #atualiza o caminho com os vetores
      out2 = np.zeros( (out.shape[0]+1, out.shape[1]) )
      out2[0] = StartPoint

      for i in range(1, out.shape[0]+1):
        out2[i] = out2[i-1] + out[i-1]
      if return_vectors == True:
        return (out2, out)
      return out2

    if EndPoint is None:
      EndPoint = self.TrashPoints[ np.random.randint(self.TrashPoints.shape[0]) ] #Escolhe um ponto final aleatóriamente
    p = EndPoint - StartPoint

    while abs(p[0]) + abs(p[1]) > self.MaxTrash: #Enquanto a distância entre os pontos não pode ser percorrida pelo caminho, atualiza
      EndPoint = self.TrashPoints[ np.random.randint(clear.TrashPoints.shape[0]) ] #Escolhe um ponto final aleatóriamente
      p = EndPoint - StartPoint

    if EndPoint[0] != StartPoint[0] and EndPoint[1] != StartPoint[1]: #Estratégia é Construir um triangulo caminhando entre os dois
      return FindPathCaseOne(StartPoint=StartPoint, EndPoint=EndPoint)

    elif (EndPoint[0] == StartPoint[0] or EndPoint[1] == StartPoint[1]) and ( (EndPoint != StartPoint).sum() != 0 ): # Estratégia da reta
      return FindPathCaseTwo(StartPoint=StartPoint, EndPoint=EndPoint)

    else: #Criar uma função que gera um caminho partindo de um ponto e iniciando em outro
      return FindPathCaseTwo(StartPoint=StartPoint)

  def GenerateSoluction(self, StartPoint:np.ndarray) -> None:
    solve = Soluction(self.TrashMesh)
    p1, p2 = self.GeneratePath(StartPoint = StartPoint, return_vectors=True)
    finish = solve.AddCaminho(p1, p2) # finish = 1 se temos lixo, 0 se não. End é o ponto onde o caminho leva
    while finish == 1: #Enquanto temos lixo, gera um caminho
      p1,p2 = self.GeneratePath(StartPoint = solve.EndPoint, return_vectors=True)
      finish = solve.AddCaminho( p1, p2)
    if self.soluctions is None:
      self.soluctions = np.array([solve])
    else:
      self.soluctions = np.append(self.soluctions, solve)

  def GenerateNSoluction(self, StartPoint:np.ndarray, n:int, show_bar=True):
    if (self.soluctions is not None) and (len(self.soluctions) != 0):
      self.soluctions = np.array([])
    if show_bar:
      for _ in tqdm(np.arange(n)):
        self.GenerateSoluction(StartPoint=StartPoint)
    else:
      for _ in np.arange(n):
        self.GenerateSoluction(StartPoint=StartPoint)

  def SaveMesh(self, path):
    np.savetxt(path, self.TrashMesh, delimiter=',')

  def LoadMesh(self, path):
    self.mesh = np.genfromtxt(path, delimiter=',')

  def SaveSoluctions():
    pass

  def LoadSoluctions():
    pass

  def CreateNGenerations(self, n:int,return_fmean=False, return_fbest=False, population=1, show_progress=True,
                         elitismo=0):
    """Cria n gerações considerando método da roleta, seleção e multação
    """
    gen_mean = np.zeros([n])
    best = np.zeros([n])
    k = len(self.soluctions)

    for g in tqdm(np.arange(n)): #para cada geração

      #Seleciona metade dos índivíuos para reprodução
      if elitismo != 0:
        newG = self.BestSoluction(elitismo)  #Elitismo
      else:
        newG = np.array([])  #Elitismo
      for i in np.arange(len(self.soluctions)//2-elitismo):
        newG = np.append(newG, self.RoletaViciada())
      # Faz o crossover dois À dois
      for i in np.arange(len(newG)):

        p1, p2 = self.Crossover(self.soluctions[i], self.soluctions[2*i+1], cutPoint=self.TrashPoints[ np.random.randint(len(self.TrashPoints))])
        if (p1 is not None) and p1.IsValid(mesh=self.mesh): newG = np.append(newG, p1)
        if (p2 is not None) and p2.IsValid(mesh=self.mesh): newG = np.append(newG, p2)

      self.soluctions = newG.copy()
      if len(self.soluctions) < k-2:
        self.GenerateNSoluction(n=int(k-len(self.soluctions)), StartPoint=self.TrashPoints[0], show_bar=False)

      #percorre a geração fazendo mutação
      for i in self.soluctions:
        pass
        #self.Mutation(i, Mutation_Chance=0.1)

      gen_mean[g] = self.__f_mean()
      best[g] = self.BestSoluction()[0].f
    if return_fmean:
      if return_fbest:
        return (gen_mean, best)
      else:
        return gen_mean

  def Mutation(self, soluction:Soluction, Mutation_Chance=0.003):
    """
    Dado um caminho, ele corta vários caminhos de modo que o ponto final e o inicial permanessa o mesmo. Mas o caminho como um todo diminua

    """
    trash = np.array([])
    trash_escolhido = self.TrashPoints[np.random.randint(len(self.TrashPoints))]
    if np.random.random() < Mutation_Chance: #Mutação acontece
      for i in np.arange(len(soluction.caminhos)): #Procura onde estão os lixos
        if np.sum(soluction.caminhos[i] == trash_escolhido) == 2:
          trash = np.append(trash, i)

      # Escolhemos dois pontos para excluir o meio
      c1 = np.random.randint( len(trash) )
      c2 = np.random.randint( len(trash) )

      p1 = soluction.caminhos[:c1].copy()
      v1 = soluction.vectors[:c1-1].copy()

      p2 = soluction.caminhos[c2:].copy()
      v2 = soluction.vectors[c2-1:].copy()

      soluction.caminhos = np.concatenate( (p1, p2) )
      soluction.vectors = np.concatenate( (v1, v2) )


  def Crossover(self, soluc1:Soluction, soluc2:Soluction, cutPoint:np.ndarray,  type=""):
    """Usa a informação de dois caminhos para montar um terceiro,
    dado dois caminho, gera o caminho mínimo entre os dois e gerar ramificações pequenas entre elas

    """
    nmax = 50 #Número máximo de tentativas
    while nmax >0:
      s1, s2 = soluc1.caminhos.copy(), soluc2.caminhos.copy()
      v1, v2 = soluc1.vectors.copy(), soluc2.vectors.copy()
      cortes1, cortes2 =  np.array([]), np.array([])
      for i in np.arange(s1.shape[0]):
        if s1[0][0] == cutPoint[0] and s1[0][1] == cutPoint[1]:
          cortes1 = np.append(cortes1, i)
      for j in np.arange(s2.shape[0]):
        if s2[0][0] == cutPoint[0] and s2[0][1] == cutPoint[1]:
          cortes1 = np.append(cortes1, j)

      i = cortes1[np.random.randint(1, cortes1.size+1)] if cortes1.size == 1 else i
      j = cortes2[np.random.randint(1, cortes2.size+1)] if cortes2.size == 1 else i

      tdepo = s1[i:].copy()
      tant = s1[:i].copy()
      v1depo = v1[i-1:].copy()
      v1ant = v1[:i-1].copy()

      t2depo = s2[j:].copy()
      t2ant = s2[:j].copy()
      v2depo = v2[j-1:].copy()
      v2ant = v2[:j-1].copy()

      s1 = np.concatenate((tant, t2depo))
      v1 = np.concatenate((v1ant, v2depo))
      s2 = np.concatenate((t2ant, tdepo))
      v2 = np.concatenate((v2ant, v1depo))

      solu1 = Soluction(mesh=self.TrashMesh.copy(), vectors=v1, caminhos=s1)
      solu2 = Soluction(mesh=self.TrashMesh.copy(), vectors=v2, caminhos=s2)
      if solu1.IsValid(mesh=self.TrashMesh) and solu2.IsValid(mesh=self.TrashMes):
        break
      else:
        nmax-=1
    if nmax  <= 0:
      return (None, None)
    else:
      return (solu1, solu2)


  def RoletaViciada(self) -> Soluction:
    """Aplica o método da roleta viciada na população atual e retorna a solução escolhida
    """
    otim = np.zeros(len(self.soluctions) )
    t=0
    for i in np.arange( len(otim)):
      otim[i] = 1/self.soluctions[i].f
      t+=1/self.soluctions[i].f
    for i in np.arange( len(otim)):
      if i==0:
        otim[i] = otim[i]/t
      else:
        otim[i] =otim[i-1]+ otim[i]/t
    t=0
    ale = np.random.random()
    while otim[t] < ale:
      t+=1
    return self.soluctions[t]

