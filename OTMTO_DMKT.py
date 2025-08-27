import copy
import openpyxl
import numpy as np
import os
import copy as cp
from CEC2017MTSO import *
from evo_operator import *
from MLP_diffusion_model import *
import torch
import random
from torch.utils.data import DataLoader
from Data_construct import MTO_data
from tasks import *
import pandas as pd
import time
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    计算两个高斯分布之间的 KL 散度。

    参数：
    mu1, sigma1: 第一个高斯分布的均值和标准差
    mu2, sigma2: 第二个高斯分布的均值和标准差
    """
    sigma1_sq = sigma1 ** 2
    sigma2_sq = sigma2 ** 2
    kl_div = np.log(sigma2 / sigma1) + (sigma1_sq + (mu1 - mu2) ** 2) / (2 * sigma2_sq) - 0.5
    return kl_div


def construct_2level_otarray(D):
    M = int(2**(np.ceil(np.log2(D+1))))
    u = int(np.log2(M))
    b = 2**(u-1)
    ot_array = np.zeros((M,(b+b-1)))

    for a in range(1,M+1):
        for k in range(1,u+1):
            b = 2**(k-1)
            ot_array[a-1][b-1] = np.floor((a-1)/(2**(u-k)))%2

    for a in range(1,M+1):
        for k in range(2,u+1):
            b = 2**(k-1)
            for s in range(1,b):
                ot_array[a-1][b+s-1] = (ot_array[a-1][s-1]+ot_array[a-1][b-1])%2

    for i in range(M):
        for j in range(D):
            if ot_array[i][j]==0:
                ot_array[i][j] = 1
            elif ot_array[i][j]==1:
                ot_array[i][j] = 2

    return  ot_array
def caculate_rad(pop,d):
    ps = len(pop)
    a = np.zeros((ps,d))
    for i in range(ps):
        a[i] = pop[i]['x'][:d]
    ctr = np.mean(a,axis=0)
    rad = 0
    for i in range(ps):
        rad = rad+np.sum((ctr-a[i])**2)
    rad = rad/ps
    return rad
def caculate_fvec(xgb,pop,d):
    nb = 5
    a = np.zeros((nb))
    for i in range(nb):
        a[i] = np.sum((pop[i]['x'][:d]-xgb[:d])**2)
    return a

def CTM_function(fvecj,xmap,target_pop,d):
    fveci = caculate_fvec(xmap,target_pop,d)
    return np.sum((fveci-fvecj)**2)

class OTMTO:
    def __init__(self,popsize, max_d,taskf1,task1d, taskf2,task2d, maxfes, KT_interval,train_interval,transfer_num, net_info,task_bound):
        self.MAXFES = maxfes
        self.fes1 = 0
        self.fes2 = 0
        self.pop_size = popsize
        self.max_d = max_d
        self.taskf1 = taskf1
        self.task1d = task1d
        self.taskf2 = taskf2
        self.task2d = task2d
        self.task1_num = 0
        self.task2_num = 0
        self.F = 0.5
        self.cr = 0.6
        self.F_best = 0.5
        self.cr_best = 0.9
        self.record_allbest_t1 = []
        self.record_allbest_t2 = []
        self.task1_best = float('inf')
        self.task1_best_position = None
        self.task2_best = float('inf')
        self.task2_best_position = None
        self.task1_otarray = construct_2level_otarray(task1d)
        self.task2_otarray = construct_2level_otarray(task2d)
        self.pot = [0.5,0.5]
        self.pcdt = [0.5,0.5]
        self.train_interval = train_interval
        self.net_info = net_info
        self.transfer_num = transfer_num
        self.KT_interval = [float(KT_interval), float(KT_interval)]
        self.transfer1 = False
        self.transfer2 = False
        self.task_bound1 = task_bound['task1']
        self.task_bound2 = task_bound['task2']
        self.archive_pop1 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.gen = 0

    def pop_init(self):
        # 初始化种群

        self.task1_pop = []
        self.task2_pop = []
        # 每个个体初始化有五个定义
        # 适应度、技能因子、任务排名(升序)、
        # 张量适应度(最优任务排名的倒数，则最大值为1)、全局最优因子(即在所有任务都是最优的个体)
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf1.function(new_pop['x'])
            new_pop['transfer'] = 0
            self.fes1 += 1
            self.task1_pop.append(new_pop)
            if new_pop['fit'] < self.task1_best:
                self.task1_best = copy.copy(new_pop['fit'])
                self.task1_best_position = copy.copy(new_pop['x'])
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf2.function(new_pop['x'])
            new_pop['transfer'] = 0
            self.fes2 += 1
            self.task2_pop.append(new_pop)
            if new_pop['fit'] < self.task2_best:
                self.task2_best = copy.copy(new_pop['fit'])
                self.task2_best_position = copy.copy(new_pop['x'])
        #对种群进行适应度排序，方便后面学习
        self.task1_pop.sort(key = lambda x:x['fit'])
        self.task2_pop.sort(key = lambda x:x['fit'])
        self.record_allbest_t1.append(copy.copy(self.task1_best))
        self.record_allbest_t2.append(copy.copy(self.task2_best))
        self.gen+=1
    def DDPM_init(self):
        self.device = get_device()
        self.pop1_model = Unet_fc(self.max_d).to(self.device)
        self.pop2_model = Unet_fc(self.max_d).to(self.device)
        self.pop1_diffusion = pop_diffusion(self.pop1_model, self.max_d,time_steps=10,loss_type='l2').to(self.device)
        self.pop2_diffusion = pop_diffusion(self.pop2_model, self.max_d, time_steps=10, loss_type='l2').to(self.device)
        self.pop1_train_ddpm = Trainer(self.pop1_diffusion,self.max_d,self.net_info['batch_size'],self.net_info['lr'],self.net_info['weight_decay'],self.net_info['num_epochs'],pop1_pop2_dataloader=self.task1_dataloader,device=self.device)
        self.pop2_train_ddpm = Trainer(self.pop2_diffusion,self.max_d,self.net_info['batch_size'],self.net_info['lr'],self.net_info['weight_decay'],self.net_info['num_epochs'],pop1_pop2_dataloader=self.task2_dataloader,device=self.device)

    def pop_data_construct(self):
        self.archive_pop1 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop1_fit = np.zeros(self.pop_size)
        self.archive_pop1_label = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2_label = np.zeros((self.pop_size, self.max_d))
        self.archive_pop2_fit = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.archive_pop1[i,:self.task1d] = self.task_bound1[0]+(self.task_bound1[1]-self.task_bound1[0])*copy.copy(self.task1_pop[i]['x'][:self.task1d])
            self.archive_pop2[i,:self.task2d] = self.task_bound2[0] + (self.task_bound2[1] - self.task_bound2[0]) * copy.copy(self.task2_pop[i]['x'][:self.task2d])

        self.task1_dataset = MTO_data(self.archive_pop1,self.archive_pop2)
        self.task2_dataset = MTO_data(self.archive_pop2,self.archive_pop1)
        self.task1_dataloader = DataLoader(self.task1_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
        self.task2_dataloader = DataLoader(self.task2_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
    def transform2list(self,pop,fit,d):
        pop_list = []
        pop_size = pop.shape[0]
        for i in range(pop_size):
            op = {}
            op['x'] = pop[i]
            op['fit'] = fit[i]
            op['transfer'] = 1
            pop_list.append(op)
        return pop_list
    def transfer_pop(self,id):
        self.pop_data_construct()
        if id ==0:
            transfer_pop1_fit = np.zeros(self.transfer_num)
            transfer_pop1 = self.pop1_train_ddpm.test_from_otherpop(self.archive_pop2[:self.transfer_num])
            transfer_pop1 = (transfer_pop1 - self.task_bound1[0]) / (self.task_bound1[1] - self.task_bound1[0])
            transfer_pop1 = np.clip(transfer_pop1, 0, 1)
            for i in range(self.transfer_num):
                transfer_pop1_fit[i] = self.taskf1.function(transfer_pop1[i])
                if transfer_pop1_fit[i] < self.task1_best:
                    self.task1_best = copy.copy(transfer_pop1_fit[i])
                self.fes1 += 1
                if (self.fes2 + self.fes1) % 200 == 0:
                    self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                    self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                if (self.fes1+self.fes2) >= self.MAXFES:
                    break
            for i in range(self.transfer_num):
                if transfer_pop1_fit[i] == 0:
                    transfer_pop1_fit[i] = float('inf')
            transfer_pop1 = self.transform2list(transfer_pop1, transfer_pop1_fit, self.task1d)
            return transfer_pop1
        elif id ==1:
            transfer_pop2_fit = np.zeros(self.transfer_num)
            transfer_pop2 = self.pop2_train_ddpm.test_from_otherpop(self.archive_pop1[:self.transfer_num])
            transfer_pop2 = (transfer_pop2 - self.task_bound2[0]) / (self.task_bound2[1] - self.task_bound2[0])
            transfer_pop2 = np.clip(transfer_pop2, 0, 1)

            for i in range(self.transfer_num):
                transfer_pop2_fit[i] = self.taskf2.function(transfer_pop2[i])
                if transfer_pop2_fit[i] < self.task2_best:
                    self.task2_best = copy.copy(transfer_pop2_fit[i])
                self.fes2 += 1
                if (self.fes2+self.fes1) % 200 == 0:
                    self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                    self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                if (self.fes1+self.fes2) >= self.MAXFES:
                    break
            for i in range(self.transfer_num):
                if transfer_pop2_fit[i] ==0:
                    transfer_pop2_fit[i] = float('inf')
            transfer_pop2 = self.transform2list(transfer_pop2, transfer_pop2_fit, self.task2d)
            return transfer_pop2
    def transfer_pop_count(self,pop):
        transfer_num = 0
        for i in range(len(pop)):
            if pop[i]['transfer'] == 1:
                transfer_num+=1
                pop[i]['transfer'] =0
        return pop, transfer_num
    def transmition2pop(self,x):
        new_pop = {}
        new_pop['x'] = x
        new_pop['transfer'] = 0
        return new_pop
    def llso_offspring_gen(self,pop,id):
        offspring = []
        for i in range(len(pop)):
            pop_id = [j for j in range(len(pop)) if j != i]
            r1, r2, r3 = np.random.choice(pop_id, 3, replace=False)
            u =DE_rand_1(pop[i]['x'],pop[r1]['x'],pop[r2]['x'],pop[r3]['x'],self.F,self.cr,self.max_d,[0,1])
            new_pop = self.transmition2pop(u)
            if id ==0:
                new_pop['fit'] = self.taskf1.function(new_pop['x'])
                self.fes1+=1
                if new_pop['fit']<self.task1_best:
                    self.task1_best = new_pop['fit']
            else:
                new_pop['fit'] = self.taskf2.function(new_pop['x'])
                self.fes2 += 1
                if new_pop['fit']<self.task2_best:
                    self.task2_best = new_pop['fit']
            offspring.append(new_pop)
            if (self.fes1+self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))
            if (self.fes1+self.fes2)>=self.MAXFES:
                return offspring

        return offspring
    def CTM(self,source_pop,target_pop):
        infe = 0
        maxINFE = 500
        inps = 50
        target_pop.sort(key = lambda x: x['fit'])
        source_pop.sort(key = lambda x: x['fit'])
        best_source = copy.deepcopy(source_pop[0]['x'])
        source_d = source_pop[0]['x'].shape[0]
        target_d = target_pop[0]['x'].shape[0]
        radj = caculate_rad(copy.deepcopy(source_pop),source_d)
        radi = caculate_rad(copy.deepcopy(target_pop),target_d)
        tvec = caculate_fvec(best_source[:source_d],copy.deepcopy(source_pop),source_d)
        fvecj = tvec* (radi/radj)

        pop = copy.deepcopy(target_pop[:inps])
        for i in range(inps):
            pop[i]['fit'] = CTM_function(fvecj,pop[i]['x'],target_pop,target_d)
        infe+=inps
        pop.sort(key = lambda x:x['fit'])
        in_best = copy.deepcopy(pop[0])
        while infe<maxINFE:
            offspring = []
            for i in range(inps):
                pop_id = [j for j in range(inps) if j != i ]
                r1, r2 = np.random.choice(pop_id, 2, replace=False)
                u = DE_rand_1(pop[i]['x'], in_best['x'], pop[r1]['x'], pop[r2]['x'], self.F_best, self.cr_best, target_d,
                              [0, 1])
                new_pop = self.transmition2pop(u)
                new_pop['fit'] = CTM_function(fvecj,new_pop['x'],target_pop,target_d)
                infe+=1
                offspring.append(new_pop)
                if new_pop['fit']<in_best['fit']:
                    in_best['fit'] = copy.copy(new_pop['fit'])
                    in_best['x'] = copy.copy(new_pop['x'])
                if infe>maxINFE:
                    break
            pop = pop+offspring
            pop.sort(key = lambda x:x['fit'])
            pop = pop[:inps]
        return in_best
    def CDT_method(self,source_pop,source_d,target_pop,target_d,id):
        reward = 0
        source_matrix = np.zeros((self.pop_size,source_d))
        target_matrix = np.zeros((self.pop_size, target_d))
        yita = 1e-6
        for i in range(len(source_pop)):
            source_matrix[i][:] = source_pop[i]['x'][:source_d]
            target_matrix[i][:] = target_pop[i]['x'][:target_d]
        source_mean = np.mean(source_matrix,axis=0)
        source_std = np.std(source_matrix,axis=1)
        target_mean = np.mean(target_matrix, axis=0)
        target_std = np.std(target_matrix, axis=1)
        pst = np.zeros((target_d,source_d))
        sim = np.zeros((target_d,source_d))

        for i in range(target_d):
            for j in range(source_d):
                sim[i][j] = 1/(kl_divergence_gaussian(target_mean[i],target_std[i],source_mean[j],source_std[j])+yita)
            sumsimi = np.sum(sim[i])
            for j in range(source_d):
                pst[i][j] = sim[i][j] /sumsimi
            pst[i] = np.cumsum(pst[i])

        target_index = []

        for i in range(target_d):
            target_index.append(np.where(pst[i]>np.random.random())[0][0])
        xcdt = np.zeros((self.max_d))
        for i in range(len(target_index)):
            xcdt[i] = np.random.normal(loc=source_mean[target_index[i]],scale=source_std[target_index[i]])

        xcdt = self.transmition2pop(xcdt)
        random_choice = np.random.randint(0,self.pop_size)
        if id ==0:
            xcdt['fit'] = self.taskf1.function(xcdt['x'])
            self.fes1 +=1
            if xcdt['fit']<self.task1_pop[random_choice]['fit']:
                self.task1_pop[random_choice] = copy.deepcopy(xcdt)
                reward = 1
            if xcdt['fit']<self.task1_best:
                self.task1_best = copy.deepcopy(xcdt['fit'])
            if (self.fes1+self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))
            self.pcdt[0] = 0.95 * self.pcdt[0] + 0.05 * reward
        else:
            xcdt['fit'] = self.taskf2.function(xcdt['x'])
            self.fes2 += 1
            if xcdt['fit']<self.task2_pop[random_choice]['fit']:
                self.task2_pop[random_choice] = copy.deepcopy(xcdt)
                reward=1
            if xcdt['fit']<self.task2_best:
                self.task2_best = copy.deepcopy(xcdt['fit'])
            if (self.fes1+self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))
            self.pcdt[1] = 0.95*self.pcdt[1]+0.05*reward






    def ot_method(self,xrki,xrki_fit,xmgb,ot_array,d,id):
        M = ot_array.shape[0]
        array_M = np.zeros((M,self.max_d))
        array_M_fit = np.zeros((ot_array.shape[0]))
        xb = np.zeros((self.max_d))
        xb_fit = float('inf')
        for i in range(M):
            for j in range(d):
                if ot_array[i][j] == 2:
                    array_M[i][j] = xrki[j]
                else:
                    array_M[i][j] = xmgb[j]
            if id ==0:
                array_M_fit[i] = self.taskf1.function(array_M[i])
                self.fes1+=1
                if (self.fes1 + self.fes2) % 200 == 0:
                    self.record_allbest_t1.append(copy.copy(self.task1_best))
                    self.record_allbest_t2.append(copy.copy(self.task2_best))
            else:
                array_M_fit[i] = self.taskf2.function(array_M[i])
                self.fes2+=1
                if (self.fes1 + self.fes2) % 200 == 0:
                    self.record_allbest_t1.append(copy.copy(self.task1_best))
                    self.record_allbest_t2.append(copy.copy(self.task2_best))
            if array_M_fit[i]<xb_fit:
                xb_fit = copy.copy(array_M_fit[i])
                xb = copy.copy(array_M[i])
        S_vector = np.zeros((2,d))
        for i in range(d):
            cum_1 = 0
            cum_1_fit = 0
            cum_2 = 0
            cum_2_fit = 0
            for j in range(M):
                if ot_array[j][i]==2:
                    cum_1+=1
                    cum_1_fit+=array_M_fit[j]
                elif ot_array[j][i]==1:
                    cum_2+=1
                    cum_2_fit+=array_M_fit[j]
            S_vector[0][i] = cum_1_fit/cum_1
            S_vector[1][i] = cum_2_fit/cum_2
        xp_vector = np.zeros((self.max_d))
        for i in range(d):
            if S_vector[0][i]<S_vector[1][i]:
                xp_vector[i] = xrki[i]
            else:
                xp_vector[i] = xmgb[i]
        if id==0:
            xp_fit = self.taskf1.function(xp_vector)
            self.fes1+=1
            if (self.fes1 + self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))
        else:
            xp_fit = self.taskf2.function(xp_vector)
            self.fes2+=1
            if (self.fes1 + self.fes2) % 200 == 0:
                self.record_allbest_t1.append(copy.copy(self.task1_best))
                self.record_allbest_t2.append(copy.copy(self.task2_best))
        if xp_fit<xb_fit:
            xot = copy.deepcopy(xp_vector)
            xot_fit = copy.deepcopy(xp_fit)

        else:
            xot = copy.deepcopy(xb)
            xot_fit = copy.deepcopy(xb_fit)
        if xot_fit<xrki_fit:
            reward = 1
        else:
            reward = 0
        xot = self.transmition2pop(xot)
        xot['fit'] = copy.copy(xot_fit)

        if id==0:
            self.pot[0] = 0.95*self.pot[0]+0.05*reward
        else:
            self.pot[1] = 0.95 * self.pot[1] + 0.05 * reward



        return xot
    def llso_optimize(self):
        transfer_interval = [1, 1]
        while (self.fes1+self.fes2)<self.MAXFES:
            if self.gen==self.train_interval:
                self.pop_data_construct()
                self.DDPM_init()
                self.pop1_train_ddpm.train()
                self.pop2_train_ddpm.train()
                self.transfer = True
                transfer_interval = [int(self.KT_interval[0]), int(self.KT_interval[1])]
            if self.gen>self.train_interval and (self.gen-self.train_interval)%self.train_interval == 0:
                self.pop_data_construct()
                self.pop1_train_ddpm.pop1_pop2_dataloader = self.task1_dataloader
                self.pop2_train_ddpm.pop1_pop2_dataloader = self.task2_dataloader
                self.pop2_train_ddpm.train()
                self.pop1_train_ddpm.train()
            if self.gen>=self.train_interval and int(self.KT_interval[0])==transfer_interval[0]:
                self.transfer_pop1= self.transfer_pop(0)
                self.task1_pop = self.task1_pop+self.transfer_pop1
                self.transfer1 = True
                transfer_interval[0] = 0
                if (self.fes1 + self.fes2) >= self.MAXFES:
                    break
            if self.gen>=self.train_interval and int(self.KT_interval[1])==transfer_interval[1]:
                self.transfer_pop2 = self.transfer_pop(1)
                self.task2_pop = self.task2_pop + self.transfer_pop2
                self.transfer2 = True
                transfer_interval[1] = 0
                if (self.fes1+self.fes2)>=self.MAXFES:
                    break
            off_task1_pop= self.llso_offspring_gen(copy.deepcopy(self.task1_pop),0)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
            off_task2_pop = self.llso_offspring_gen(copy.deepcopy(self.task2_pop), 1)
            if (self.fes1+self.fes2)>=self.MAXFES:
                break
            self.task1_pop= self.task1_pop+off_task1_pop
            self.task2_pop = self.task2_pop+off_task2_pop
            self.task1_pop.sort(key = lambda x:x['fit'])
            self.task2_pop.sort(key = lambda x:x['fit'])
            self.task1_pop = self.task1_pop[:self.pop_size]
            self.task2_pop = self.task2_pop[:self.pop_size]

            if np.random.random()<self.pot[0]:
                xmgb = self.CTM(copy.deepcopy(self.task2_pop), copy.deepcopy(self.task1_pop))
                random_choice = np.random.randint(0,self.pop_size)
                xot = self.ot_method(copy.deepcopy(self.task1_pop[random_choice]['x']),copy.deepcopy(self.task1_pop[random_choice]['fit']),xmgb['x'],self.task1_otarray,self.task1d,0)
                if xot['fit']<self.task1_pop[random_choice]['fit']:
                    self.task1_pop[random_choice] = copy.deepcopy(xot)
            if np.random.random()<self.pcdt[0]:
                self.CDT_method(copy.deepcopy(self.task2_pop),self.task2d,copy.deepcopy(self.task1_pop),self.task1d,0)
            if np.random.random() < self.pot[1]:
                xmgb = self.CTM(copy.deepcopy(self.task1_pop), copy.deepcopy(self.task2_pop))
                random_choice = np.random.randint(0, self.pop_size)
                xot = self.ot_method(copy.deepcopy(self.task2_pop[random_choice]['x']),
                                     copy.deepcopy(self.task2_pop[random_choice]['fit']), xmgb['x'], self.task2_otarray,
                                     self.task2d, 1)
                if xot['fit'] < self.task2_pop[random_choice]['fit']:
                    self.task2_pop[random_choice] = copy.deepcopy(xot)
            if np.random.random() < self.pcdt[1]:
                self.CDT_method(copy.deepcopy(self.task1_pop), self.task1d, copy.deepcopy(self.task2_pop), self.task2d,
                                1)
            if  self.transfer1:
                self.transfer1 = False
            if self.transfer2:
                self.transfer2 = False
            self.gen+=1
            transfer_interval[0] = transfer_interval[0] + 1
            transfer_interval[1] = transfer_interval[1] + 1
        best_1 = self.task1_best
        best_2 = self.task2_best
        return best_1,best_2,self.record_allbest_t1,self.record_allbest_t2



task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_2020 = [Benchmark1(),Benchmark2(),Benchmark3(),Benchmark4(),Benchmark5(),Benchmark6(),Benchmark7(),Benchmark8(),Benchmark9(),Benchmark10()]
task_2020_name = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5','Benchmark6','Benchmark7','Benchmark8','Benchmark9','Benchmark10']
task_bound = {'CIHS':{'task1':[-100,100],'task2':[-50,50]},'CIMS':{'task1':[-50,50],'task2':[-50,50]},\
              'CILS':{'task1':[-50,50],'task2':[-500,500]},'PIHS':{'task1':[-50,50],'task2':[-100,100]},\
              'PIMS':{'task1':[-50,50],'task2':[-50,50]},'PILS':{'task1':[-50,50],'task2':[-0.5,0.5]},\
              'NIHS':{'task1':[-50,50],'task2':[-50,50]},'NIMS':{'task1':[-100,100],'task2':[-0.5,0.5]},\
              'NILS':{'task1':[-50,50],'task2':[-500,500]},}
task_d = [[50,50],[50,50],[50,50],[50,50],[50,50],[50,25],[50,50],[50,50],[50,50]]
net_info = {}
net_info['num_epochs'] = 100
net_info['lr'] = 1e-2
net_info['batch_size'] = 100
net_info['weight_decay'] = 1e-5

test_task = ['WCCI2020MTSO']
if __name__ == '__main__':
    for test in test_task:
        if test == 'WCCI2020MTSO':
            record_matrix = np.zeros((30,len(task_2020)*2))
            filename_with_suffix = __file__
            filename_without_suffix = os.path.basename(filename_with_suffix)
            filename = os.path.splitext(filename_without_suffix)[0]+'_5_25_50'+test
            ori_path = './independent_data/' + filename
            xls_savepath = './record_data/' + filename + '.xlsx'
            if not os.path.exists(ori_path):
                os.makedirs(ori_path, exist_ok=True)
            task_num = len(task_2020)
            mfea_record = openpyxl.Workbook()
            sheet = mfea_record.active
            sheet['A1'] = 'test_task'
            sheet['B1'] = 'task1'
            sheet['C1'] = 'task2'
            best_avg_array = np.zeros((task_num, 2))
            for i in range(task_num):
                task1_20_bestall = []
                task2_20_bestall = []
                path = ori_path+'/' + task_2020_name[i] + '.txt'
                task1,task2 = task_2020[i]
                task1_bound = [task1.Low, task1.High]
                task2_bound = [task2.Low, task2.High]
                task_b = {}
                task_b['task1'] = task1_bound
                task_b['task2'] = task2_bound
                for j in range(30):
                    mfllso = OTMTO(100,50,task1,50,task2,50,1e+05,5,50,25,net_info,task_b)
                    mfllso.pop_init()
                    best_t1,best_t2,bestall_t1,bestall_t2 = mfllso.llso_optimize()
                    task1_20_bestall.append(bestall_t1)
                    task2_20_bestall.append(bestall_t2)
                    record_matrix[j][i*2] = best_t1
                    record_matrix[j][(i * 2+1)] = best_t2
                    best_avg_array[i][0] += best_t1
                    best_avg_array[i][1] += best_t2
                    f_file = open(path, 'a+')
                    f_file.write(f'{best_t1} {best_t2}\n')
                    f_file.close()
                task_xlsx_path = ori_path+'/' + task_2020_name[i] + '.xlsx'
                task1_20_bestall = np.array(task1_20_bestall)
                task2_20_bestall = np.array(task2_20_bestall)
                task1_20_bestall = pd.DataFrame(task1_20_bestall)
                task2_20_bestall = pd.DataFrame(task2_20_bestall)
                with pd.ExcelWriter(task_xlsx_path,engine='openpyxl') as writer:
                    task1_20_bestall.to_excel(writer,index=False, sheet_name='Task1')
                    task2_20_bestall.to_excel(writer,index=False, sheet_name='Task2')
            best_avg_array /= 30
            for i in range(task_num):
                sheet.append([task_2020_name[i], best_avg_array[i][0], best_avg_array[i][1]])
            mfea_record.save(xls_savepath)
            record_matrix = pd.DataFrame(record_matrix)
            pd_path = ori_path+'/' + 'WCCI2020MTSO.xlsx'
            record_matrix.to_excel(pd_path,index=False)