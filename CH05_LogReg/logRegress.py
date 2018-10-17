from math import *
from numpy import *
#加载数据 原始数据前两列是特征 第三列是标签
#在特征列表的第一列加了个1.0 方便计算
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=200
    weights=ones((n,1))
    #print(dataMatrix)
    #input()
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        #h = dot(dataMatrix, weights)
        #h=sigmoid(h)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
#画出数据集 和最佳拟合曲线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dateArr=array(dataMat)
    n=shape(dateArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dateArr[i,1])
            ycord1.append(dateArr[i,2])
        else:
            xcord2.append(dateArr[i,1])
            ycord2.append(dateArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
#随机梯度上升法
def stoGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights
#改进后的随机梯度上升法
#第一点改进：alpha每次迭代的时候都会调整
#二：通过随机选取样本更新回归系数
def stoGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)

    for j in range(numIter):
        #这里原书是range(m),下面删除变量的时候会报错，python3range返回的是range对象而不是数组
        dataIndex = list(range(m))
        for i in range(m):
            #alpha每次迭代时需要调整
            alpha=4/(1.0+j+i)+0.01
            #随机选取更新
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def classsifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        #print(line)
        currLine=line.strip().split('\t')
        #print(currLine)
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stoGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classsifyVector (array(lineArr),trainWeights )) !=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate is %f"%errorRate)
    return errorRate
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d the av error rate is%f"%(numTests,errorSum/float(numTests)))
if __name__ == '__main__':
    #dateArr,labelMat=loadDataSet()
    #weights=stoGradAscent1(array(dateArr),labelMat)
    #plotBestFit(weights)
    multiTest()
