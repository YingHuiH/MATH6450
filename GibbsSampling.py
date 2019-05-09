# -*- coding: utf-8 -*-
"""

@author: erint_000
"""

import os
import pandas as pd
import random
import math
from scipy.stats import uniform, gamma,invwishart
import numpy as np
import matplotlib.pyplot as plt 
from numpy.linalg import cholesky
import seaborn as sns
import copy

def get_simulate_data():
    
    Lambda_m = np.mat(np.zeros((3,9)))
    Lambda_m[0,0] = Lambda_m[1,3] = Lambda_m[2,6] = 1
    Lambda_m[0,1] = Lambda_m[0,2] = Lambda_m[1,4] = Lambda_m[1,5] = Lambda_m[2,7] =Lambda_m[2,8]= 0.8 
    Lambda_m= Lambda_m.T
    
    Lambda_w_1 = Lambda_w_2 = 0.3
    Phi_11 = Phi_22 = 0.5
    Phi_12 = 0.5
    
    Psi_e = np.diag([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    Psi_delta = 0.1
    
    
    sampleNo = 40
    # 生成Xi - 二维正态分布
    np.random.seed(111)
    
    mu = np.array([[0, 0]])
    Sigma = np.array([[Phi_11, Phi_12], [0, Phi_22]])
    R = cholesky(Sigma)
    
    Xi = np.dot(np.random.randn(sampleNo, 2), R) + mu

    delta = np.random.normal(0, Psi_delta, sampleNo )
    
    Eta = np.mat(Xi) * np.mat([[Lambda_w_1],[Lambda_w_2]]) + np.mat(delta).T
    print(Xi[0],delta[0],Eta[0])
    
    
    Epsilon_0 = np.random.normal(0, np.sqrt(Psi_e[0][0]), sampleNo )
    Epsilon_1 = np.random.normal(0, np.sqrt(Psi_e[1][1]), sampleNo )
    Epsilon_2 = np.random.normal(0, np.sqrt(Psi_e[2][2]), sampleNo )
    Epsilon_3 = np.random.normal(0, np.sqrt(Psi_e[3][3]), sampleNo )
    Epsilon_4 = np.random.normal(0, np.sqrt(Psi_e[4][4]), sampleNo )
    Epsilon_5 = np.random.normal(0, np.sqrt(Psi_e[5][5]), sampleNo )
    Epsilon_6 = np.random.normal(0, np.sqrt(Psi_e[6][6]), sampleNo )
    Epsilon_7 = np.random.normal(0, np.sqrt(Psi_e[7][7]), sampleNo )
    Epsilon_8 = np.random.normal(0, np.sqrt(Psi_e[8][8]), sampleNo )

    y_simulate = np.mat(np.zeros(( sampleNo,9)))
    for i in range(0,sampleNo):
        Omega = [np.float(Eta[i]),Xi[i][0],Xi[i][1]]
        Epsilon = np.mat([ [Epsilon_0[i]], [Epsilon_1[i]], 
                           [Epsilon_2[i]], [Epsilon_3[i]], 
                           [Epsilon_4[i]], [Epsilon_5[i]],
                           [Epsilon_6[i]], [Epsilon_7[i]],
                           [Epsilon_8[i]] ])
        y = Lambda_m * np.mat(Omega).T + Epsilon

        y_simulate[i] = y.T
    
    return y_simulate





if __name__ == '__main__':
    

    y = get_simulate_data()  #获取simulated数据
    
    #y_tmp = pd.DataFrame(y) #画图观察数据关系
    #print(y_tmp.describe())
    #print(y_tmp.corr())
    #sns.heatmap(y_tmp.corr(),annot=False, vmax=1,vmin = -1, xticklabels= False, yticklabels= False, square=True, cmap="seismic") #"YlGnBu"  "RdBu"

    

    # Gibbs sampler
    E=5000  
    BURN_IN= 200
    
    Sample_No = 40  # 设置样本量
    Num_LatentVar = 3  # 潜变量数量
    Num_ExLatentVar = 2
    Num_ObservedVar = np.shape(y)[1]  # 自动获取可观测变量数量
    
    
    # Initialize the chain
    Omega = np.mat(np.zeros((Sample_No, Num_LatentVar)))
    Lambda = np.mat(np.zeros(( Num_ObservedVar, Num_LatentVar )))
    Lambda[0,0] = Lambda[3,1] = Lambda[6,2] = 0.3  # 设置初始值
    Lambda[1,0] = Lambda[2,0] = Lambda[4,1] = Lambda[5,1] = Lambda[7,2] =Lambda[8,2]= 0.3  # 设置初始值
    
    
    Psi_e = np.mat(np.eye(Num_ObservedVar, Num_ObservedVar)*0.1) # 生成一个对角阵
    alpha0_ek = 9 # 设置初始值
    beta0_ek = 4 # 设置初始值
    for i in range(0,Num_ObservedVar):
        psi_ek = 1 / random.gammavariate(alpha0_ek, beta0_ek)
        Psi_e[i,i] = psi_ek
    
    Lambda_Pi = np.mat(np.zeros((Num_LatentVar-Num_ExLatentVar,Num_LatentVar-Num_ExLatentVar)))
    Lambda_Gamma = np.mat(np.zeros((Num_LatentVar-Num_ExLatentVar,Num_ExLatentVar)))
    Lambda_Gamma[0,0] = 0.4
    Lambda_Gamma[0,1] = 0.4 # 设置初始值
    #Lambda_w = np.hstack( Lambda_Pi, Lambda_Gamma)    # 稀疏矩阵会报错，用下面的代替
    Lambda_w = np.column_stack(( Lambda_Pi, Lambda_Gamma ))
    
        
    Phi = np.mat(np.eye(Num_ExLatentVar, Num_ExLatentVar)*0.5)    
    Phi[0,0] = Phi[1,1] = 0.5   # 设置初始值
    Phi[0,1] = 0.5      # 设置初始值
       
    Psi_delta = np.mat(np.eye(Num_LatentVar-Num_ExLatentVar, Num_LatentVar-Num_ExLatentVar)*0.5)    
    alpha0_dk = 9
    beta0_dk = 4
    for i in range(0,np.shape(Psi_delta)[0]):
        psi_dk = 1 / random.gammavariate(alpha0_dk, beta0_dk)
        Psi_delta[i,i] = psi_dk
        
        
    # Store the samples
    chain_lambda_00 =np.array([0.]*(E-BURN_IN))
    chain_lambda_31 =np.array([0.]*(E-BURN_IN))
    chain_lambda_62 =np.array([0.]*(E-BURN_IN))
    chain_lambda_10 =np.array([0.]*(E-BURN_IN))
    chain_lambda_20 =np.array([0.]*(E-BURN_IN))
    chain_lambda_41 =np.array([0.]*(E-BURN_IN))
    chain_lambda_51 =np.array([0.]*(E-BURN_IN))
    chain_lambda_72 =np.array([0.]*(E-BURN_IN))
    chain_lambda_82 =np.array([0.]*(E-BURN_IN))
    
    chain_psi_e_1 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_2 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_3 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_4 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_5 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_6 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_7 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_8 = np.array([0.]*(E-BURN_IN))
    chain_psi_e_9 = np.array([0.]*(E-BURN_IN))
    
    chain_lambda_w_1 = np.array([0.]*(E-BURN_IN))
    chain_lambda_w_2 = np.array([0.]*(E-BURN_IN))
    
    chain_phi_00 = np.array([0.]*(E-BURN_IN))
    chain_phi_11 = np.array([0.]*(E-BURN_IN))
    chain_phi_01 = np.array([0.]*(E-BURN_IN))
    
    chain_psi_delta = np.array([0.]*(E-BURN_IN))

    
    # 单位阵，备用
    I = np.mat(np.eye(Num_LatentVar-Num_ExLatentVar ,Num_LatentVar-Num_ExLatentVar ,dtype=int))
    
    Lambda_0 = copy.deepcopy(Lambda)
    
    H_0yk = np.mat(np.eye(Num_LatentVar ,Num_LatentVar ,dtype=int))
    H_0yk = 5 * H_0yk  #设置初始值
    
    rho = 7 #设置初始值
    R0 = np.mat(np.eye(Num_ExLatentVar, Num_ExLatentVar))    
    

    H_0wk = np.mat(np.eye(Num_LatentVar ,Num_LatentVar ,dtype=int))
    H_0wk = 5*H_0wk   #设置初始值
    Lambda_w0 = copy.deepcopy(Lambda_w)   
    
    
    for e in range(E):
            print("At iteration "+str(e))    
            # update Omega
            for i in range(0,Sample_No): 
                Lambda_Pi_0 = I-Lambda_Pi
                sigma_w_00 = Lambda_Pi_0.I * (Lambda_Gamma * Phi * Lambda_Gamma.T + Psi_delta)  * (Lambda_Pi_0.I).T  
                sigma_w_01 = Lambda_Pi_0.I * Lambda_Gamma * Phi
                sigma_w_11 = Phi * Lambda_Gamma.T * (Lambda_Pi_0.I).T
                sigma_w_12 = Phi
                
                sigma_w_0 = np.column_stack(( sigma_w_00, sigma_w_01 ))
                sigma_w_1 = np.column_stack(( sigma_w_11, sigma_w_12 ))
                sigma_w = np.row_stack(( sigma_w_0, sigma_w_1 ))
            
                mu = (sigma_w.I + Lambda.T * Psi_e.I * Lambda).I * Lambda.T * Psi_e.I * y[i].T
                SS = (sigma_w.I + Lambda.T * Psi_e.I * Lambda).I
                R = cholesky(SS) 
                Omega[i] = np.dot(np.random.randn(1, len(mu)), R) + mu.T #注意mu要转置！因为此时它是列向量
            
        
            # update Psi_ek
            for k in range(0, Num_ObservedVar): #
                Ak = (H_0yk.I + Omega.T * Omega ).I
                ak = Ak * (H_0yk.I * Lambda_0[k].T + Omega.T * (y.T[k]).T) # (y.T[k]).T 实际上是Yk
                beta_ek = beta0_ek + 1/2 * ((y.T[k]) * (y.T[k]).T - ak.T * Ak.I * ak + Lambda_0[k] * H_0yk.I * Lambda_0[k].T)
                psi_ek = 1 / random.gammavariate( Sample_No/2 + alpha0_ek, beta_ek)
                Psi_e[k,k] = psi_ek
        
            
            # updata Lambda
            for k in range(0, Num_ObservedVar):
                ck = np.array([0.]*Num_LatentVar)
                Omega_k_xx = np.array([0.]*Sample_No)
                Lambda_0k_xx = []
                rk = 0
                for c in range(0,Num_LatentVar):
                    if Lambda_0[k,c] != 0:
                        rk += 1
                        ck[c] = 1
                        Omega_k_xx=np.row_stack((Omega_k_xx,Omega.T[c]))   
                        Lambda_0k_xx.append(Lambda_0[k,c])
                
                Omega_k_xx = np.delete(Omega_k_xx, 0, 0)  # delete first row of A
                Lambda_0k_xx = np.mat(Lambda_0k_xx)
                y_k_xx =  y.T[k] #因为认为固定的参数都为0，所以减的项目=0
                H_0yk_xx = np.mat(np.eye(rk ,rk ,dtype=int))
                H_0yk_xx = 10 * H_0yk_xx 
                #
                Ak_xx = (H_0yk_xx.I + Omega_k_xx * Omega_k_xx.T ).I
                ak_xx = Ak_xx * (H_0yk_xx.I * Lambda_0k_xx.T + Omega_k_xx * (y_k_xx).T) # (y.T[k]).T 实际上是Yk
                mu = ak_xx
                SS = Psi_e[k,k] * Ak_xx
                R = cholesky(SS) 
                
                
                tmp = np.dot(np.random.randn(1, len(mu)), R) + mu.T #注意mu要转置！因为此时它是列向量
                #只更新非零的量
                ttt = 0 
                for z in range(0,Num_LatentVar):
                    if Lambda_0[k,z] != 0:
                        #print(k,z,Lambda_0[k])
                        Lambda[k,z] = tmp [0,ttt]
                        ttt += 1
            
        
            # update Phi
            Omega_1 = Omega.T[0 :Num_LatentVar - Num_ExLatentVar]
            Omega_2 = Omega.T[Num_LatentVar - Num_ExLatentVar :Num_LatentVar]

            df_set = Sample_No + rho  
            scale_set = Omega_2 * Omega_2.T + R0.I
            Phi  = invwishart.rvs(df = df_set, scale = scale_set)
            
            # update Psi_delta_k
            for k in range(0, Num_LatentVar - Num_ExLatentVar ): #
                Awk = (H_0wk.I + Omega.T * Omega ).I
                awk = Awk * (H_0wk.I * Lambda_w0[k].T + Omega.T * Omega_1[k].T ) 
                beta_d_k = beta0_dk + 1/2 * (( Omega_1[k] ) * (Omega_1[k].T) - awk.T * Awk.I * awk + Lambda_w0[k] * H_0wk.I * Lambda_w0[k].T)
                psi_dk = 1 / random.gammavariate( Sample_No/2 + alpha0_dk, beta0_dk)
                Psi_delta[k,k] = psi_dk    
            
            # update Lambda_w
            for k in range(0, Num_LatentVar - Num_ExLatentVar ):
                ck = np.array([0.]*Num_LatentVar)
                Omega_w_k_xx = np.array([0.]* Sample_No)
                Lambda_w0k_xx = []
                rk = 0
                for c in range(0,Num_LatentVar):
                     if Lambda_w0[k,c] != 0:
                            rk += 1
                            ck[c] = 1
                            Omega_w_k_xx=np.row_stack((Omega_w_k_xx,Omega.T[c]))   
                            Lambda_w0k_xx.append(Lambda_w0[k,c])
                    
                Omega_w_k_xx = np.delete(Omega_w_k_xx, 0, 0)  # delete first row of A
                Lambda_w0k_xx = np.mat(Lambda_w0k_xx)
                y_w_k_xx =  y.T[k] #因为认为固定的参数都为0，所以减的项目=0
                H_w_0yk_xx = np.mat(np.eye(rk ,rk ,dtype=int))
                H_w_0yk_xx = 10 * H_w_0yk_xx  #week prior 
                
                Awk_xx = (H_w_0yk_xx.I + Omega_w_k_xx * Omega_w_k_xx.T ).I
                awk_xx = Awk_xx * (H_w_0yk_xx.I * Lambda_w0k_xx.T + Omega_w_k_xx * (Omega_1[k].T) ) 
                mu = awk_xx
                SS = Psi_delta[k,k] * Awk_xx
                R = cholesky(SS) 
                tmp = np.dot(np.random.randn(1, len(mu)), R) + mu.T #注意mu要转置！因为此时它是列向量
                #只更新非零的量
                ttt = 0 
                for q in range(0,Num_LatentVar):
                    if Lambda_w0[k,q] != 0:
                        #print(k,z,Lambda_0[k])
                        Lambda_w[k,q] = tmp [0,ttt]
                        ttt += 1                
                
        
            # store
            if e>=BURN_IN:
                chain_lambda_10[e-BURN_IN]=Lambda[1,0]
                chain_lambda_31[e-BURN_IN]=Lambda[3,1]
                chain_lambda_62[e-BURN_IN]=Lambda[6,2]
                chain_lambda_20[e-BURN_IN]=Lambda[2,0]
                chain_lambda_41[e-BURN_IN]=Lambda[4,1]
                chain_lambda_51[e-BURN_IN]=Lambda[5,1]
                chain_lambda_72[e-BURN_IN]=Lambda[7,2]
                chain_lambda_82[e-BURN_IN]=Lambda[8,2]

                chain_lambda_w_1[e-BURN_IN]=Lambda_w[0,1]
                chain_lambda_w_2[e-BURN_IN]=Lambda_w[0,2]
   
                chain_psi_e_1[e-BURN_IN]=Psi_e[0,0]
                chain_psi_e_2[e-BURN_IN]=Psi_e[1,1]
                chain_psi_e_3[e-BURN_IN]=Psi_e[2,2]
                chain_psi_e_4[e-BURN_IN]=Psi_e[3,3]
                chain_psi_e_5[e-BURN_IN]=Psi_e[4,4]
                chain_psi_e_6[e-BURN_IN]=Psi_e[5,5]
                chain_psi_e_7[e-BURN_IN]=Psi_e[6,6]
                chain_psi_e_8[e-BURN_IN]=Psi_e[7,7]
                chain_psi_e_9[e-BURN_IN]=Psi_e[8,8]
                

                chain_phi_00[e-BURN_IN]=Phi[0,0]
                chain_phi_11[e-BURN_IN]=Phi[1,1]
                chain_phi_01[e-BURN_IN]=Phi[0,1]
                
                chain_psi_delta[e-BURN_IN]=Psi_delta

                   
	def draw_line_plot(chain,real_value,title_set):
		
		os.chdir('D:\\figures')
		plt.figure()
		plt.plot(chain) 
		plt.plot(np.array([real_value]*len(chain))) # 添加真值的参考线
		plt.title(title_set )
		plt.ylim((0,2))
		
		plt.savefig(str(title_set+'.png'),dpi = 500)
		plt.show()
		return            


	# 查看MCMC收敛情况
	draw_line_plot(chain_lambda_31,1.0,'lambda_31-line')  
	draw_line_plot(chain_lambda_62,1.0,'lambda_61-line')
		
	draw_line_plot(chain_lambda_10, 0.8,'Lambda.10-line')
	draw_line_plot(chain_lambda_20,0.8,'lambda_20-line')
	draw_line_plot(chain_lambda_41,0.8,'lambda_41-line')
	draw_line_plot(chain_lambda_51,0.8,'lambda_51-line')
	draw_line_plot(chain_lambda_72,0.8,'lambda_72-line')
	draw_line_plot(chain_lambda_82,0.8,'lambda_82-line')

	draw_line_plot(chain_lambda_w_1,0.3,'chain_lambda_w_1-line')
	draw_line_plot(chain_lambda_w_2,0.3,'chain_lambda_w_2-line')

	def draw_hist_plot(chain,real_value,title_set):
		
		os.chdir('D:\\figures')
		plt.figure()
		num_bins = 50# the histogram of the data
		n, bins, patches = plt.hist(chain, num_bins, normed=1, facecolor='blue', alpha=0.5)
		describe = 'Mean='+ str(np.mean(chain[200:]))[0:6] + '  Std=' + str(np.std(chain[200:]))[0:6]  #给图上添加统计数据，以备查用
		print(describe) 
		plt.xlabel(( describe ))  
		plt.ylabel('Frequency')
		plt.title(title_set )
		plt.savefig(str(title_set+'.png'),dpi = 500)
		plt.show()
		return   
		
	draw_hist_plot(chain_lambda_31,1.0,'lambda_31-hist')

	draw_hist_plot(chain_lambda_62,1.0,'lambda_61-hist')
		
	draw_hist_plot(chain_lambda_10, 0.8,'Lambda.10-hist')
	draw_hist_plot(chain_lambda_20,0.8,'lambda_20-hist')
	draw_hist_plot(chain_lambda_41,0.8,'lambda_41-hist')
	draw_hist_plot(chain_lambda_51,0.8,'lambda_51-hist')
	draw_hist_plot(chain_lambda_72,0.8,'lambda_72-hist')
	draw_hist_plot(chain_lambda_82,0.8,'lambda_82-hist')

	draw_hist_plot(chain_lambda_w_1,0.3,'chain_lambda_w_1-hist')
	draw_hist_plot(chain_lambda_w_1,0.3,'chain_lambda_w_2-hist')

