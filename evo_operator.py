import numpy as np
import random as rd
import copy
def SBX_crossover(a,b,D,mu,task_bound):
    mu = mu
    lbound = task_bound[0]
    ubound = task_bound[1]
    child1 = np.zeros(D)
    child2 = np.zeros(D)
    u = np.random.uniform(0,1,size=D)
    cf = np.zeros(D)
    for i in range(D):
        if u[i]<=0.5:
            cf[i] = (2*u[i])**(1/(mu+1))
        else:
            cf[i] = (2*(1-u[i]))**(-1/(mu+1))
        tmp = 0.5 * ((1 - cf[i]) * a[i] + (1 + cf[i]) * b[i])
        child1[i] = tmp
        tmp = 0.5 * ((1 + cf[i]) * a[i] + (1 - cf[i]) * b[i])
        child2[i] = tmp
    child1 = np.clip(child1,0,1)
    child2 = np.clip(child2, 0, 1)
    """for i in range(D):
        r1 = np.random.rand()
        if r1 <= 0.5:
            eta_c = 10
            y1 = min(a[i], b[i])
            y2 = max(a[i], b[i])
            if (y1 - lbound) > (ubound - y2):
                beta = 1 + 2 * (ubound - y2) / (y2 - y1)
            else:
                beta = 1 + 2 * (y1 - lbound) / (y2 - y1)
            expp = eta_c + 1
            beta = 1 / beta
            alpha = 2 - beta ** expp
            r2 = np.random.rand()
            if r2 <= (1 / alpha):
                alpha = alpha * r2
                expp = 1 / (eta_c + 1)
                betaq = alpha ** (expp)
            else:
                alpha = 1 / (2 - alpha * r2)
                expp = 1 / (eta_c + 1)
                betaq = alpha ** expp
            aa = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            bb = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            aa = max(aa, lbound)
            bb = max(bb, lbound)
            if np.random.rand() > 0.5:
                child1[i] = min(aa, ubound)
                child2[i] = min(bb, ubound)
            else:
                child2[i] = min(aa, ubound)
                child1[i] = min(bb, ubound)
        else:
            child1[i] = copy.copy(a[i])
            child2[i] = copy.copy(b[i])"""
    return child1, child2


def poly_mutation(pop,D,mu,task_bound):
    """for i in range(D):
        if rd.random() < pm:
            eta_m = 5
            if pop[i] > lbound:
                if (pop[i] - lbound) < (ubound - pop[i]):
                    delta = (pop[i] - lbound) / (ubound - lbound)
                else:
                    delta = (ubound - pop[i]) / (ubound - lbound)
                e_x = 1 / (eta_m + 1)
                temp_rand = rd.random()
                if temp_rand <= 0.5:
                    temp = 1 - delta
                    val = 2 * temp_rand + (1 - 2 * temp_rand) * (temp ** (eta_m + 1))
                    deltaq = val ** e_x - 1
                else:
                    temp = 1 - delta
                    val = 2 * (1 - temp_rand) + 2 * (temp_rand - 0.5) * (temp ** (eta_m + 1))
                    deltaq = 1 - val ** e_x
                pop[i] = pop[i] + deltaq * (ubound - lbound)
                pop[i] = max(pop[i], lbound)
                pop[i] = min(pop[i], ubound)
            else:
                pop[i] = rd.random() * (ubound - lbound) + lbound
    return pop"""
    # 要用多项式变异
    mu = mu
    lbound = task_bound[0]
    ubound = task_bound[1]
    offsp = copy.deepcopy(pop)
    for i in range(D):
        rand1 = np.random.rand()
        if rand1 < (1/D):
            rand2 = np.random.rand()
            if rand2 < 0.5:
                tmp1 = (2 * rand2) ** (1 / (1 + mu)) - 1
                tmp2 = pop[i] + tmp1*pop[i]
                offsp[i] = tmp2
            else:
                tmp1 = 1-(2*(1-rand2))**(1/(1+mu))
                tmp2 = pop[i] + tmp1*(1-pop[i])
                offsp[i] = tmp2
        else:
            offsp[i] = pop[i]
    offsp = np.clip(offsp,0,1)
    return offsp

def DE_rand_1(x, p1, p2, p3, F, cr, d,bound):
    lb = bound[0]
    ub = bound[1]
    v = p1 + F * ( p2 - p3 )
    u = DE_crossover( x, v, cr, d)
    u = np.clip(u, lb,ub)
    return u
def DE_rand_2(x, p1, p2, p3, p4, p5, F1, F2, cr, d,bound):
    lb = bound[0]
    ub = bound[1]
    v = p1 + F1 * ( p2 - p3 ) + F2*(p4-p5)
    u = DE_crossover( x, v, cr, d)
    u = np.clip(u, lb,ub)
    return u
def DE_crossover(x,v,cr,d):
    u = copy.deepcopy(v)
    k = np.random.randint(0, d)
    for i in range(d):
        if np.random.rand()<cr or i==k:
            u[i] = v[i]
        else:
            u[i] = x[i]
    return u

