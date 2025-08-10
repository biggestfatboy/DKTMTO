import copy
import math
import openpyxl
import numpy as np
import os
import copy as cp
from CEC2017MTSO import *
from evo_operator import *
from MLP_diffusion_model import *
import random
from tasks import *
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Data_construct import MTO_data
import argparse
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class DDPM_MTO:
    def __init__(self,popsize,taskf1, task1_d, taskf2, task2_d, KT_interval,train_interval,transfer_num, maxgen, net_info,task_bound):
        self.MAXgen = maxgen
        self.fes1 = 0
        self.fes2 = 0
        self.pop_size = popsize
        self.max_d = max(task1_d,task2_d)
        self.taskf1 = taskf1
        self.taskf2 = taskf2
        self.task1_d = task1_d
        self.task2_d = task2_d
        self.record_allbest_t1 = []
        self.record_ktinterval_t1 = []
        self.record_ktinterval_t2 = []
        self.record_allbest_t2 = []
        self.task1_best = float('inf')
        self.task2_best = float('inf')
        self.gen = 0
        self.F = 0.3
        self.CR = [0.6,0.6]
        self.pop2_evor = 'DE'
        self.pop1_evor = 'DE'
        self.KT_interval = [float(KT_interval),float(KT_interval)]
        self.archive_pop1 = np.zeros((self.pop_size,self.max_d))
        self.archive_pop2 = np.zeros((self.pop_size, self.max_d))
        self.net_info = net_info
        self.transfer_num = transfer_num
        self.transfer = False
        self.task_bound1 = task_bound['task1']
        self.task_bound2 = task_bound['task2']
        self.MAXfes = 1e05
        self.fes = 0
        self.train_interval = train_interval
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
            self.archive_pop1[i,:self.task1_d] = self.task_bound1[0]+(self.task_bound1[1]-self.task_bound1[0])*copy.copy(self.task1_pop[i]['x'][:self.task1_d])
            self.archive_pop2[i,:self.task2_d] = self.task_bound2[0] + (self.task_bound2[1] - self.task_bound2[0]) * copy.copy(self.task2_pop[i]['x'][:self.task2_d])


        self.task1_dataset = MTO_data(self.archive_pop1,self.archive_pop2)
        self.task2_dataset = MTO_data(self.archive_pop2,self.archive_pop1)
        self.task1_dataloader = DataLoader(self.task1_dataset, batch_size=self.net_info['batch_size'], shuffle=True)
        self.task2_dataloader = DataLoader(self.task2_dataset, batch_size=self.net_info['batch_size'], shuffle=True)

    def cr_assign(self):
        for i in range(self.pop_size):
            if random.random()<self.CR_p[0]:
                self.task1_pop[i]['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            else:
                self.task1_pop[i]['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            if random.random()<self.CR_p[1]:
                self.task2_pop[i]['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            else:
                self.task2_pop[i]['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes

    def pop_init(self):
        # 初始化种群
        self.task1_pop = []
        self.task2_pop = []
        self.task1_totalfit = 0
        self.task2_totalfit = 0
        t1_pop = np.zeros((self.pop_size, self.max_d))
        t1_pop_fit = np.zeros(self.pop_size)
        t2_pop = np.zeros((self.pop_size, self.max_d))
        t2_pop_fit = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf1.function(new_pop['x'])
            new_pop['transfer'] = 0
            if random.random()<0.5:
                new_pop['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            else:
                new_pop['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            t1_pop[i,:self.task1_d] = new_pop['x'][:self.task1_d]
            t1_pop_fit[i] = new_pop['fit']
            self.fes1 += 1
            self.task1_pop.append(new_pop)
            if new_pop['fit'] < self.task1_best:
                self.task1_best = copy.copy(new_pop['fit'])
            if self.fes1 ==1 :
                self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))

        for i in range(self.pop_size):
            new_pop = {}
            new_pop['x'] = np.random.random(self.max_d)
            new_pop['fit'] = self.taskf2.function(new_pop['x'])
            new_pop['transfer'] = 0
            if random.random()<0.5:
                new_pop['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            else:
                new_pop['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            t2_pop[i,:self.task2_d] = copy.copy(new_pop['x'][:self.task2_d])
            t2_pop_fit[i] = copy.copy(new_pop['fit'])
            self.fes2 += 1
            self.task2_pop.append(new_pop)
            if new_pop['fit'] < self.task2_best:
                self.task2_best = copy.copy(new_pop['fit'])
            if self.fes2==1:
                self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
        #对种群进行适应度排序，方便后面学习
        self.task1_pop.sort(key = lambda x:x['fit'])
        self.task2_pop.sort(key = lambda x:x['fit'])
        for i in range(self.pop_size):
            self.archive_pop1[i,:self.task1_d] = self.task1_pop[i]['x'][:self.task1_d]
            self.archive_pop2[i,:self.task2_d] = self.task2_pop[i]['x'][:self.task2_d]
        if (self.fes2 + self.fes1) % 200 == 0:
            self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
            self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
            self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))
            self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
        self.pop_data_construct()
        self.DDPM_init()
        self.cr_assign()
        self.gen+=1
    def task1_generation(self):
        offspring=[]
        pop = copy.deepcopy(self.task1_pop)
        pop_id = [i for i in range(len(pop))]
        for i in range(len(pop)):
            pop_id = [j for j in range(len(pop)) if j != i ]
            r1,r2,r3 = np.random.choice(pop_id,3,replace=False)
            r1 = copy.deepcopy(pop[r1])
            r2 = copy.deepcopy(pop[r2])
            r3 = copy.deepcopy(pop[r3])
            op = {}
            op['x'] = DE_rand_1(pop[i]['x'],r1['x'],r2['x'],r3['x'],self.F,pop[i]['cr'],self.task1_d,[0,1])
            op['x'] = np.clip(op['x'],0,1)
            op['fit'] = self.taskf1.function(op['x'])
            op['cr'] = copy.copy(pop[i]['cr'])
            op['transfer'] = 0
            self.task1_pop[i]['cr'] =1
            if self.task1_best > op['fit']:
                self.task1_best = copy.deepcopy(op['fit'])
            offspring.append(op)
            self.fes1+=1
            if (self.fes2 + self.fes1) % 200 == 0:
                self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))
                self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
            if (self.fes1+self.fes2)>=self.MAXfes:
                break

        return offspring
    def task2_generation(self):
        offspring=[]
        pop = copy.deepcopy(self.task2_pop)
        for i in range(len(pop)):

            pop_id = [j for j in range(len(pop)) if j != i]
            r1, r2, r3 = np.random.choice(pop_id, 3, replace=False)
            r1 = copy.deepcopy(pop[r1])
            r2 = copy.deepcopy(pop[r2])
            r3 = copy.deepcopy(pop[r3])
            op = {}
            op['x'] = DE_rand_1(pop[i]['x'], r1['x'], r2['x'], r3['x'], self.F, pop[i]['cr'],
                                self.task2_d, [0, 1])
            op['fit'] = self.taskf2.function(op['x'])

            op['transfer'] = 0
            op['cr'] = copy.copy(pop[i]['cr'])
            self.task2_pop[i]['cr'] = 1
            if self.task2_best > op['fit']:
                self.task2_best = op['fit']
            offspring.append(op)
            self.fes2 += 1
            if (self.fes2 + self.fes1) % 200 == 0:
                self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))
                self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
            if (self.fes1+self.fes2) >= self.MAXfes:
                break

        return offspring
    def transform2list(self,pop,fit,d):
        pop_list = []
        pop_size = pop.shape[0]
        for i in range(pop_size):
            op = {}
            op['x'] = pop[i]
            op['fit'] = fit[i]
            op['transfer'] = 1
            if random.random() < 0.5:
                op['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
            else:
                op['cr'] = 0.6+0.3*(self.fes1+self.fes2)/self.MAXfes
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
                    self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))
                    self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
                if (self.fes1+self.fes2) >= self.MAXfes:
                    break
            for i in range(self.transfer_num):
                if transfer_pop1_fit[i] == 0:
                    transfer_pop1_fit[i] = float('inf')
            transfer_pop1 = self.transform2list(transfer_pop1, transfer_pop1_fit, self.task1_d)
            return transfer_pop1
        elif id ==1:
            transfer_pop2_fit = np.zeros(self.transfer_num)
            transfer_pop2 = self.pop2_train_ddpm.test_from_otherpop(self.archive_pop1[:self.transfer_num])
            transfer_pop2 = (transfer_pop2 - self.task_bound2[0]) / (self.task_bound2[1] - self.task_bound2[0])
            transfer_pop2 = np.clip(transfer_pop2, 0, 1)

            for i in range(self.transfer_num):
                transfer_pop2_fit[i] = self.taskf2.function(transfer_pop2[i])
                a = transfer_pop2_fit[i]
                b = self.taskf1.function(transfer_pop2[i])
                if transfer_pop2_fit[i] < self.task2_best:
                    self.task2_best = copy.copy(transfer_pop2_fit[i])
                self.fes2 += 1
                if (self.fes2+self.fes1) % 200 == 0:
                    self.record_allbest_t1.append(copy.deepcopy(self.task1_best))
                    self.record_allbest_t2.append(copy.deepcopy(self.task2_best))
                    self.record_ktinterval_t1.append(copy.deepcopy(self.KT_interval[0]))
                    self.record_ktinterval_t2.append(copy.deepcopy(self.KT_interval[1]))
                if (self.fes1+self.fes2) >= self.MAXfes:
                    break
            for i in range(self.transfer_num):
                if transfer_pop2_fit[i] ==0:
                    transfer_pop2_fit[i] = float('inf')
            transfer_pop2 = self.transform2list(transfer_pop2, transfer_pop2_fit, self.task2_d)
            return transfer_pop2
    def optimize_tasks(self):
        transfer_interval = [1, 1]
        while (self.fes1+self.fes2)<self.MAXfes:
            if self.gen==self.train_interval: #第一次训练
                self.pop_data_construct()
                self.pop1_train_ddpm.pop1_pop2_dataloader = self.task1_dataloader
                self.pop2_train_ddpm.pop1_pop2_dataloader = self.task2_dataloader
                self.pop1_train_ddpm.train()
                self.pop2_train_ddpm.train()
                self.transfer = True
                transfer_interval = [int(self.KT_interval[0]), int(self.KT_interval[1])]
            if self.gen>self.train_interval and (self.gen-self.train_interval)%self.train_interval == 0:#每隔g代更新一次网络
                self.pop_data_construct()
                self.pop1_train_ddpm.pop1_pop2_dataloader = self.task1_dataloader
                self.pop2_train_ddpm.pop1_pop2_dataloader = self.task2_dataloader
                self.pop2_train_ddpm.train()
                self.pop1_train_ddpm.train()

            if self.gen>=self.train_interval and int(self.KT_interval[0])==transfer_interval[0] :#任务2迁移到任务1
                self.transfer_pop1= self.transfer_pop(0)
                self.task1_pop = self.task1_pop+self.transfer_pop1
                self.transfer1 = True
                transfer_interval[0] = 0
                if (self.fes1 + self.fes2) >= self.MAXfes:
                    break
            if self.gen>=self.train_interval and int(self.KT_interval[1])==transfer_interval[1]:#任务1迁移到任务2
                self.transfer_pop2 = self.transfer_pop(1)
                self.task2_pop = self.task2_pop + self.transfer_pop2
                self.transfer2 = True
                transfer_interval[1] = 0
                if (self.fes1+self.fes2)>=self.MAXfes:
                    break
            off1 = self.task1_generation() #自身进化
            if (self.fes1 + self.fes2) >= self.MAXfes:
                break
            off2 = self.task2_generation()#自身进化
            if (self.fes1 + self.fes2) >= self.MAXfes:
                break

            self.task1_pop = self.task1_pop + off1
            self.task2_pop = self.task2_pop + off2
            self.task1_pop.sort(key = lambda x:x['fit'])
            self.task2_pop.sort(key = lambda x:x['fit'])

            self.task1_pop = self.task1_pop[:self.pop_size]
            self.task2_pop = self.task2_pop[:self.pop_size]
            self.gen+=1
            transfer_interval[0] = transfer_interval[0] + 1
            transfer_interval[1] = transfer_interval[1] + 1
            self.cr_assign()
        return self.task1_best,self.record_allbest_t1,self.task2_best,self.record_allbest_t2

def record_point(mat,points,i):
    for j in range(len(points)):
        mat[i][j] = points[j]

    return mat




parser = argparse.ArgumentParser(description='DDMTO参数')
parser.add_argument("--transfer_num",type=int,default=25,help="迁移个数")
parser.add_argument("--train_interval",type=int,default=50,help="网络训练间隔")
parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
parser.add_argument("--ti",type=int,default=5,help="learning rate")
args = parser.parse_args()
print(args)
task_fc = ['CIHS','CIMS','CILS','PIHS','PIMS','PILS','NIHS','NIMS','NILS']
task_2020 = [Benchmark1(),Benchmark2(),Benchmark3(),Benchmark4(),Benchmark5(),Benchmark6(),Benchmark7(),Benchmark8(),Benchmark9(),Benchmark10()]
task_2020_name = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5','Benchmark6','Benchmark7','Benchmark8','Benchmark9','Benchmark10']
task_d = [[50,50],[50,50],[50,50],[50,50],[50,50],[50,25],[50,50],[50,50],[50,50]]
test_task = ['WCCI2020MTSO']
net_info = {}
net_info['num_epochs'] = 100
net_info['lr'] = args.lr
net_info['batch_size'] = 100
net_info['weight_decay'] = 1e-5
independent_runs = 30
if __name__ == '__main__':
    for test in test_task:
        if test == 'WCCI2020MTSO':
            record_matrix = np.zeros((30,len(task_2020)*2))
            filename_with_suffix = __file__
            filename_without_suffix = os.path.basename(filename_with_suffix)
            filename = os.path.splitext(filename_without_suffix)[0]+'_'+str(args.transfer_num)+'_'+str(args.train_interval)+'_'+str(int(10000*args.lr))+'_'+str(args.ti)+'_'+test
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
                task1_20_bestall = np.zeros((independent_runs,501))
                task2_20_bestall = np.zeros((independent_runs,501))
                path = ori_path+'/' + task_2020_name[i] + '.txt'
                task1,task2 = task_2020[i]
                task1_bound = [task1.Low,task1.High]
                task2_bound = [task2.Low, task2.High]
                task_b = {}
                task_b['task1'] = task1_bound
                task_b['task2'] = task2_bound
                for j in range(30):
                    ddpm_mto = DDPM_MTO(100,task1,50,task2,50,args.ti,args.train_interval,args.transfer_num,500,net_info,task_b)
                    ddpm_mto.pop_init()
                    best_t1,bestall_t1,best_t2,bestall_t2 = ddpm_mto.optimize_tasks()
                    task1_20_bestall = record_point(task1_20_bestall,bestall_t1,j)
                    task2_20_bestall = record_point(task2_20_bestall, bestall_t2, j)
                    record_matrix[j][i*2] = best_t1
                    record_matrix[j][(i * 2+1)] = best_t2
                    best_avg_array[i][0] += best_t1
                    best_avg_array[i][1] += best_t2
                    f_file = open(path, 'a+')
                    f_file.write(f'{best_t1} {best_t2}\n')
                    f_file.close()
                task_xlsx_path = ori_path+'/' + task_2020_name[i] + '.xlsx'
                task_xlsx_frequency_path = ori_path + '/' + task_2020_name[i] + '_ktfrequency.xlsx'

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
