import numpy as np

# def iniciar_temperatura(Tk=0.1, L0, Rmin=0.9):
#     u = u0
#     RA = 0
#     while RA < Rmin:
#         u, RA = cadena_markov(u, L0)
#         Tk *= beta
#     return Tk


knapsack = np.asarray([[23,31,29,44,53,38,63,85,89,82],
                       [92,57,49,68,60,43,67,84,87,72]])

w_max = 165

def evaluar(u):
    fu = np.sum(u *knapsack[1])
    w = np.sum(u * knapsack[0])
    if w > w_max:
        fu += (w_max-w)*3
    fu = max(fu,0)
    return fu, w

def generar_vecino(u):
    idx = np.random.randint(0, u.shape[0]-1)
    vecino = np.copy(u)
    vecino[idx] = np.logical_not(vecino[idx].astype(np.bool)).astype(vecino.dtype)
    return vecino

print(evaluar(np.asarray([1,0,1,1,0,1,0,1,1,0])))