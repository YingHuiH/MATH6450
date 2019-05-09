# -*- coding: utf-8 -*-
"""
@author: Yinghui
"""

import os
import numpy as np
import numpy.linalg as la #  power method，and svd
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn import preprocessing #W矩阵标准化
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st
import scipy.optimize as opt   # 求解极大似然函数
import matplotlib.pylab as plt
import random
import pylab

def get_max_eigenvec(A):
    #  Power method
    eps = 1e-8 # Precision of eigenvalue

    Xk = np.mat(np.array([0.1]*np.shape(A)[0])).T
    Yk = A * Xk
    
    tolerance  = 1e-6
    value = 1  # 保证循环有效
    Yk_list = []
    while np.abs(value) > tolerance :
        Yk = A.T * Xk
        Xk_1 = Yk / np.sqrt(Yk.T * Yk)
        rk = Xk_1.T * (A.T * Xk_1)
        
        value = np.linalg.norm(A.T * Xk - np.multiply(rk , Xk))
        Yk_list.append(Yk)
        Xk = Xk_1
    
    return rk, Xk
'''
测试是否获取了最大的特征向量
    B = np.array([1, 2, 3, 2, 4, 5, 3, 5, -1, 3, 5, -1, 3, 5, -1]).reshape((3, 5))
    eigval, eigvec = get_max_eigenvalue(B)
    print np.linalg.eigh( np.mat(B.T) *np.mat(B) )
    print eigvec
    print np.max(np.linalg.eigvals(A)) ,eigval # should be very close!!
'''    
  
def get_top_k_eigvec(B,top_k):
    eigval_top = []
    eigvec_top = []
    A =np.mat(B.T) *np.mat(B)
    target_M = A
    for i in range(top_k):
        eigval, eigvec = get_max_eigenvec(target_M)
        eigval_top.append(eigval)
        eigvec_top.append(eigvec.T)
        target_M = target_M - float(eigval) * eigvec  * eigvec.T #update  
    return eigval_top , eigvec_top
'''
# 测试是否准确获取了 top k eigenvec and eigenval:
    B = np.array([1, 2, 3, 2, 4, 5, 3, 5, -1, 3, 5, -1, 3, 5, -1]).reshape((3, 5))
    top_k_set = 3

    eigvalue_sq , eigvector = get_top_k_eigvec(B, top_k_set)    
    
    print np.linalg.eigh( (np.mat(B.T) *np.mat(B))  )
    print eigvector
    print np.max(np.linalg.eigvals(np.mat(B.T) *np.mat(B))) 
    print eigvalue_sq # should be very close!!   
    eigvector = np.mat(np.array(eigvector)).T
    A  =  B * eigvector
    print B[0],   eigvector * A[0].T # two should be equal. which means we can recover B[i] by  eigvector * A[i]
'''


def get_data():
    # 
    W = np.load('D:/W.npy') #事先单独标准化的genotype matrix
    os.chdir('D:\\')
    # 读取phenotype数据    
    Y_data = []
    with open('y.csv') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        Y_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到G_data中
            Y_data.append(row[1:]) #去掉R转存时候的序号

    
    Y_data = [[float(x) for x in row] for row in Y_data]  # 将数据从string形式转换为float形式
    
    
    Y_data = np.array(Y_data)  # 将list数组转化成array数组便于查看数据结构
    print(Y_data.shape)
    
    # 读取SNPs名称
    SNPs_data = []
    with open('SNPs.csv') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        SNPs_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到G_data中
            SNPs_data.append(row[1: ]) 
    
    
    # 查看数据的结构和内容
    print(Y_data[0:10][0:10])
    print(SNPs_data[0:10])

    
     # 读取PCA数据  事先单独计算的主成分得分数据
    PCA_data = []
    Stop_set = 10  # top K principal component score 
    with open('PCA_PCs.csv') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        PCA_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到G_data中
            PCA_data.append(row[1: Stop_set+1 ]) #去掉R转存时候的序号

    
    PCA_data = [[float(x) for x in row] for row in PCA_data]  # 将数据从string形式转换为float形式
    PCA_data = np.array(PCA_data)
    tmp = np.mat(([1.]*np.shape(PCA_data)[0])).T  # 在PCA基础上增加一个单位向量
    X_data = np.column_stack((PCA_data,tmp))

    return Y_data ,X_data, W




def find_delta(y, X, W):
    n = np.shape(W)[0]   #获取观测数量
    U,S,VT = la.svd(W)   #矩阵分解供后续函数直接调用
    y = y.reshape((n,1)) 
    Uy = U.T.dot(y)
    UX = U.T.dot(X)

    def cal_LL(delta, y, X, W, Uy,UX ): #计算likelihood，输出为当前参数设置下的似然
        n = np.shape(W)[0]
        Sd = S + delta
        UXS = UX / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UX.shape[1]), (Sd.itemsize,0))
        UyS = Uy / Sd
        
        beta = sp.dot((UXS.T.dot(UX)).I , UXS.T.dot(Uy))   #计算给定delta时的beta hat
             
        r = 0
        for i in range(0,n):
            r += np.square(float(Uy[i] - UX[i].dot(beta))) / Sd[i]

        sigma2 = r / n
        nLL = 0.5 * ( np.log(Sd).sum() + n * (np.log(2.0 * np.pi * sigma2) + 1))
        #print delta, nLL
        return nLL
    
    def beta_estimate(delta, y, X, W, Uy,UX ):  #用于得到optimal delta值之后计算对应的beta估计
        Sd = S + delta
        UXS = UX / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UX.shape[1]), (Sd.itemsize,0))
        UyS = Uy / Sd
        beta = sp.dot((UXS.T.dot(UX)).I , UXS.T.dot(Uy))
        return beta
    def delta_u_estimate(delta,beta, y, X, W, Uy,UX ):  #用于得到optimal delta值之后计算对应的sigma u的估计
        n = np.shape(W)[0]
        Sd = S + delta
        r = 0
        for i in range(0,n):
            r += np.square(float(Uy[i] - UX[i].dot(beta))) / Sd[i]

        delta_u = 1.0 /n * r
        
        return delta_u
    
    x0 = 0.5
    mycoeffs = (y, X, W, Uy,UX)
    minimum =  opt.minimize(cal_LL,x0,args=mycoeffs )   #优化LL（delta）

    print('min:', minimum.x)   
    
    
    beta_hat = beta_estimate( minimum.x,  y, X, W, Uy,UX)  #计算MLE下的beta
    
    deta_u_hat = delta_u_estimate( minimum.x, beta_hat, y, X, W, Uy,UX) #计算MLE下的sigma u
    
    print 'beta:',beta_hat
    print 'delta:',deta_u_hat
    #min = minimize1D(f=f, nGrid=nGridH2, minval=0.001, maxval=1.0 )  #绘制图像
    #x = np.linspace(0,1,100) 
    #plt.plot(x,cal_LL(x,y, X, W, Uy,UX)) 
    #plt.plot(minimum.x,cal_LL(minimum.x, y, X, W, Uy,UX),'ro') 
    #plt.show()

    return  deta_u_hat, minimum.x



if __name__ == '__main__':

    
    os.chdir('D:\\')

    Y,X,W = get_data()  #获取数据
    find_delta(Y[:,0],X,W)   #计算对phenotype 1 的heritability 
    find_delta(Y[:,1],X,W)	 #计算对phenotype 2 的heritability 
    find_delta(Y[:,2],X,W)
    find_delta(Y[:,3],X,W)
    
    plt.subplot(4,1,1)   #查看所有观测的phenotype的分布情况 
    plt.scatter(range(Y.shape[0]),Y[:,0],s=0.3,alpha = 0.5,)
    plt.subplot(4,1,2)
    plt.scatter(range(Y.shape[0]),Y[:,1],c='green',s=0.3,alpha = 0.5)
    plt.subplot(4,1,3)
    plt.scatter(range(Y.shape[0]),Y[:,2],c='orange',s=0.3,alpha = 0.5)
    plt.subplot(4,1,4)
    plt.scatter(range(Y.shape[0]),Y[:,3],c='red',s=0.3,alpha = 0.5)
    plt.xlabel('ID of Individual')
    plt.show()
    

    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    pylab.suptitle("$K$")
    pylab.imshow(W, cmap=pylab.gray(), vmin=0, vmax=1)
    pylab.show()   

    N  = 500   # 观察Fast LMM在数据量变化的情况下，估计结果的变化
    find_delta(Y[0:N_individual,1],X[0:N_individual,:],W[0:N_individual,0:N_individual])
    find_delta(Y[0:N_individual,2],X[0:N_individual,:],W[0:N_individual,0:N_individual])
    find_delta(Y[0:N_individual,3],X[0:N_individual,:],W[0:N_individual,0:N_individual])
    
    


    def FisherInfo_beta(sigma_u2,delta_opt):
        I_beta = - 1.0 / sigma_u2 *  X.T * (W - np.mat(np.eye(W.shape[0])* delta_opt) ) * X
        std_beta = I_beta.I
        print np.diag(std_beta)
        
        return np.mat(np.diag(np.diag(std_beta)))
    
    Cov_beta = FisherInfo_beta(0.2259,3.3849)   #计算给定参数时的beta 的fisher information，用于std估计

    


	tmp = []
	target = 3  #设定当前分析的是哪个phenotype
	for i in range(100):   #sampling的方式，查看std
		index = random.sample(range(5123),5000)    
		Y_slice = Y[index,target]
		X_slice = X[index,:]
		W_slice = W[index,:]
		W_slice = W_slice [:,index]
		deta_u_hat ,x = find_delta( Y_slice , X_slice, W_slice)  
		tmp.append([deta_u_hat ,x])

	tmp_deta_u = []
	tmp_h = []
	for j in range(10):
		tmp_deta_u.append(tmp[j][0]) 
		tmp_h.append( 1.0 / (float(tmp[j][1]) + 1) )

	print 'phenotype=', target, 'sigma u2',np.std(tmp_deta_u),'h2',np.std(tmp_h)



    