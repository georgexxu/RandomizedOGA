# In this version of OGA with random dictionaries, we use QMC to evaluate the loss function. 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
from scipy.sparse import linalg
from pathlib import Path
import itertools
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 
pi = torch.tensor(np.pi,dtype=torch.float64)
torch.set_default_dtype(torch.float64)


class model(nn.Module):
    """ ReLU k shallow neural network
    Parameters: 
    input size: input dimension
    hidden_size1 : number of hidden layers 
    num_classes: output classes 
    k: degree of relu functions
    """
    def __init__(self, input_size, hidden_size1, num_classes,k = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, num_classes,bias = False)
        self.k = k 
    def forward(self, x):
        u1 = self.fc2(F.relu(self.fc1(x))**self.k)
        return u1


def generate_relu_dict4D_QMC(s,N0):
    # Sob = torch.quasirandom.SobolEngine(dimension =4, scramble= True, seed=None) 
    # samples = Sob.draw(N0).double() 

    # for i in range(s-1):
    #     samples = torch.cat([samples,Sob.draw(N0).double()],0)

    # Monte Carlo 
    samples = torch.rand(s*N0,4) 

    T =torch.tensor([[pi,0,0,0],[0,pi,0,0],[0,0,2*pi,0],[0,0,0,2*2]])
    shift = torch.tensor([0,0,0,-2])
    samples = samples@T + shift 

    f1 = torch.zeros(s*N0,1) 
    f2 = torch.zeros(s*N0,1)
    f3 = torch.zeros(s*N0,1)
    f4 = torch.zeros(s*N0,1)
    f5 = torch.zeros(s*N0,1)

    f1[:,0] = torch.cos(samples[:,0]) 
    f2[:,0] = torch.sin(samples[:,0]) * torch.cos(samples[:,1])
    f3[:,0] = torch.sin(samples[:,0]) * torch.sin(samples[:,1]) * torch.cos(samples[:,2])
    f4[:,0] = torch.sin(samples[:,0]) * torch.sin(samples[:,1]) * torch.sin(samples[:,2])  
    f5[:,0] = samples[:,3]

    Wb_tensor = torch.cat([f1,f2,f3,f4,f5],1) # N x 4 
    return Wb_tensor

def adjust_neuron_position(my_model, dims = 3):

    def create_mesh_grid(dims, pts):
        mesh = torch.tensor(list(itertools.product(pts, repeat=dims)))
        vertices = mesh.reshape(len(pts) ** dims, -1) 
        return vertices
    counter = 0 
    # positions = torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
    pts = torch.tensor([0.,1.])
    positions = create_mesh_grid(dims,pts) 
    neuron_num = my_model.fc1.bias.size(0)
    for i in range(neuron_num): 
        w = my_model.fc1.weight.data[i:i+1,:]
        b = my_model.fc1.bias.data[i]
    #     print(w,b)
        values = torch.matmul(positions,w.T) # + b
        left_end = - torch.max(values)
        right_end = - torch.min(values)
        offset = (right_end - left_end)/50
        if b <= left_end + offset/2 : 
            b = torch.rand(1)*(right_end - left_end - offset) + left_end + offset/2 
            my_model.fc1.bias.data[i] = b 
        if b >= right_end - offset/2 :
            if counter < (dims+1):
#                 print("here")
                counter += 1
            else: # (d + 1) or more 
                b = torch.rand(1)*(right_end - left_end - offset) + left_end + offset/2 
                my_model.fc1.bias.data[i] = b 
    return my_model

def MonteCarlo_Sobol_dDim_weights_points(M ,d = 4):
    Sob_integral = torch.quasirandom.SobolEngine(dimension =d, scramble= False, seed=None) 
    integration_points = Sob_integral.draw(M).double() 
    integration_points = integration_points.to(device)
    weights = torch.ones(M,1).to(device)/M 
    return weights, integration_points 



def minimize_linear_layer_explicit_assemble(model,target,weights, integration_points,solver="direct"):
    """
    calls the following functions (dependency): 
    1. GQ_piecewise_2D
    input: the nn model containing parameter 
    1. define the loss function  
    2. take derivative to extract the linear system A
    3. call the cg solver in scipy to solve the linear system 
    output: sol. solution of Ax = b
    """
    start_time = time.time() 
    w = model.fc1.weight.data 
    b = model.fc1.bias.data 
    basis_value_col = F.relu(integration_points @ w.t()+ b)**(model.k) 
    weighted_basis_value_col = basis_value_col * weights 
    jac = weighted_basis_value_col.t() @ basis_value_col 
     
    rhs = weighted_basis_value_col.t() @ (target(integration_points)) 
    print("assembling the matrix time taken: ", time.time()-start_time) 
    start_time = time.time()    
    if solver == "cg": 
        sol, exit_code = linalg.cg(np.array(jac.detach().cpu()),np.array(rhs.detach().cpu()),tol=1e-12)
        sol = torch.tensor(sol).view(1,-1)
    elif solver == "direct": 
#         sol = np.linalg.inv( np.array(jac.detach().cpu()) )@np.array(rhs.detach().cpu())
        sol = (torch.linalg.solve( jac.detach(), rhs.detach())).view(1,-1)
    elif solver == "ls":
        sol = (torch.linalg.lstsq(jac.detach().cpu(),rhs.detach().cpu(),driver='gelsd').solution).view(1,-1)
        # sol = (torch.linalg.lstsq(jac.detach(),rhs.detach()).solution).view(1,-1) # gpu/cpu, driver = 'gels', cannot solve singular
    print("solving Ax = b time taken: ", time.time()-start_time)
    return sol 

def OGAL2FittingReLU4D_QMC(my_model,target,s,N0,num_epochs, M, k =1, linear_solver = "direct"): 
    
    """ Orthogonal greedy algorithm using 1D ReLU dictionary over [-pi,pi]
    Parameters
    ----------
    my_model: 
        nn model 
    target: 
        target function
    num_epochs: int 
        number of training epochs 
    integration_intervals: int 
        number of subintervals for piecewise numerical quadrature 

    Returns
    -------
    err: tensor 
        rank 1 torch tensor to record the L2 error history  
    model: 
        trained nn model 
    """
    #Todo Done
    # samples for QMC integral
    start_time = time.time()
    # Sob_integral = torch.quasirandom.SobolEngine(dimension =4, scramble= False, seed=None) 
    # integration_points = Sob_integral.draw(M).double() 
    # integration_points = integration_points.to(device)
    weights, integration_points = MonteCarlo_Sobol_dDim_weights_points(M ,d = 4) 
    print("generate sob sequence:", time.time() - start_time) 

    err = torch.zeros(num_epochs+1)
    if my_model == None: 
        func_values = target(integration_points)
        num_neuron = 0

        list_b = []
        list_w = []

    else: 
        func_values = target(integration_points) - my_model(integration_points).detach()
        bias = my_model.fc1.bias.detach().data
        weights = my_model.fc1.weight.detach().data
        num_neuron = int(bias.size(0))

        list_b = list(bias)
        list_w = list(weights)
    
    # initial error Todo Done

    func_values_sqrd = func_values*func_values
    # print(func_values_sqrd.size())
    # print(gw_expand.size() ) 

    err[0]= torch.mean(func_values_sqrd)**0.5
    all_start_time = time.time()
    
    solver = linear_solver
    print("using linear solver: ",solver)
    for i in range(num_epochs): 
        relu_dict_parameters = generate_relu_dict4D_QMC(s,N0)  
        print("epoch: ",i+1, end = '\t')
        if num_neuron == 0: 
            func_values = target(integration_points)
        else: 
            func_values = target(integration_points) - my_model(integration_points).detach()

        start_time = time.time() 
        basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[:,0:4].T ) - relu_dict_parameters[:,4])**k).T # uses broadcasting
        output = torch.abs(torch.matmul(basis_values,func_values))/M # 
        neuron_index = torch.argmax(output.flatten())
        print("argmax time taken, ", time.time() - start_time)
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:4]) # 
        list_b.append(-relu_dict_parameters[neuron_index,4])
        num_neuron += 1
        my_model = model(4,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        #Todo 
        start_time = time.time() 
        sol = minimize_linear_layer_explicit_assemble(my_model,target,weights,integration_points, solver)
        print("\t\t time taken minimize linear layer: ",time.time() - start_time) 
        my_model.fc2.weight.data[0,:] = sol[:]

        func_values = target(integration_points) - my_model(integration_points).detach()
        func_values_sqrd = func_values*func_values

        #Todo Done 
        err[i+1]= torch.mean(func_values_sqrd)**0.5
        print("current error: ",err[i+1]) 
    print("total duration: ",time.time() - all_start_time)
    return err, my_model

def print_convergence_order(err, neuron_num_exponent): 

    neuron_nums = [2**j for j in range(2,neuron_num_exponent)]
    err_list = [err[i] for i in neuron_nums ] 

    print("neuron num \t\t error \t\t order")
    for i, item in enumerate(err_list):
        if i == 0: 
            print(neuron_nums[i], end = "\t\t")
            print(item, end = "\t\t")
            print("*")
        else: 
            print(neuron_nums[i], end = "\t\t")
            print(item, end = "\t\t") 
            print(np.log(err_list[i-1]/err_list[i])/np.log(2))


if __name__ == "__main__": 
    def target(x):
        ## Gaussian function in 4d 
        d = 4 
        cn =   7.03/d 
        return torch.exp(-torch.sum( cn**2 * (x - 0.5)**2,dim = 1, keepdim = True)) 

    experiment_label = "ex2"
    for k in [1,4]: 

        for s in [2**4, 2**5]: 
            print()
            print() 
            exponent = 9  
            num_epochs=  2**exponent 
            N0 = 2**5 
            M = 2**19 # around 50w 
            print(M)
            my_model = None 
            err, my_model = OGAL2FittingReLU4D_QMC(my_model,target,s,N0,num_epochs, M, k = k, linear_solver = "direct")

            filename = experiment_label + "_err_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(k,s * N0,num_epochs)
            torch.save(err,filename)
            filename = experiment_label + "_model_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(k,s * N0,num_epochs)
            torch.save(my_model.state_dict(),filename) 
    
            print_convergence_order(err,exponent) 

