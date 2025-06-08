# %%
# In this version of OGA with random dictionaries, we use QMC to evaluate the loss function. 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import os 
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
    

def show_convergence_order(err_l2,exponent,dict_size,k,d, filename,write2file = False):
    
    if write2file:
        file_mode = "a" if os.path.exists(filename) else "w"
        f_write = open(filename, file_mode)
    
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    l2_order = -1/2-(2*k + 1)/(2*d)
    if write2file:
        f_write.write('dictionary size: {}\n'.format(dict_size))
        f_write.write("neuron num \t\t error \t\t order{} \t\t h10 error \\ order \n".format(l2_order))
    print("neuron num \t\t error \t\t order")
    for i, item in enumerate(err_list):
        if i == 0: 
            print("{} \t\t {:.6f} \t\t *  \n".format(neuron_nums[i],item ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t * \t\t \n".format(neuron_nums[i],item ))
        else: 
            print("{} \t\t {:.6f} \t\t {:.6f} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2) ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t {} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2) ))
    if write2file:     
        f_write.write("\n")
        f_write.close()


def show_convergence_order_latex(err_l2,exponent,k=1,d=1): 
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    l2_order = -1/2-(2*k + 1)/(2*d)
    print("neuron num  & \t $\\|u-u_n \\|_{{L^2}}$ & \t order $O(n^{{{:.2f}}})$  \\\\ \\hline \\hline ".format(l2_order))
    for i, item in enumerate(err_list):
        if i == 0: 
            print("{} \t\t & {:.6f} &\t\t *  \\\ \hline  \n".format(neuron_nums[i],item) )   
        else: 
            print("{} \t\t &  {:.3e} &  \t\t {:.2f} \\\ \hline  \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2) ) )


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



def generate_relu_dict4D(N_list):
    
    N = np.prod(N_list) 

    grid_indices = [np.linspace(0,1,N_item,endpoint=False) for N_item in N_list]
    grid = np.meshgrid(*grid_indices,indexing='ij')
    grid_coordinates = np.column_stack([axis.ravel() for axis in grid]) 
    samples = torch.tensor(grid_coordinates) 

    T =torch.tensor([[pi,0,0,0],[0,pi,0,0],[0,0,2*pi,0],[0,0,0,2*2]]) # 2 * sqrt(d)
    shift = torch.tensor([0,0,0,-2])
    samples = samples@T + shift 

    f1 = torch.zeros(N,1) 
    f2 = torch.zeros(N,1)
    f3 = torch.zeros(N,1)
    f4 = torch.zeros(N,1)
    f5 = torch.zeros(N,1)

    f1[:,0] = torch.cos(samples[:,0]) 
    f2[:,0] = torch.sin(samples[:,0]) * torch.cos(samples[:,1])
    f3[:,0] = torch.sin(samples[:,0]) * torch.sin(samples[:,1]) * torch.cos(samples[:,2])
    f4[:,0] = torch.sin(samples[:,0]) * torch.sin(samples[:,1]) * torch.sin(samples[:,2])  
    f5[:,0] = samples[:,3]

    Wb_tensor = torch.cat([f1,f2,f3,f4,f5],1) # N x 4 
    return Wb_tensor.to(device)


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
    return Wb_tensor.to(device)

def generate_relu_dict4plusD_sphere(dim, s,N0): # 
    samples = torch.randn(s*N0,dim +1) 
    samples = samples/samples.norm(dim=1,keepdim=True)  
    Wb = samples 
    return Wb 

def adjust_neuron_position(my_model, dims = 3):

    def create_mesh_grid(dims, pts):
        mesh = torch.tensor(list(itertools.product(pts,repeat=dims)))
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
    return weights.to(device), integration_points.to(device) 

def compute_l2_error(u_exact,my_model,M,batch_size_2,weights,integration_points): 
    err = 0 
    if my_model == None: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            func_values = u_exact(integration_points[jj:end_index,:])
            err += torch.sum(func_values**2 * weights[jj:end_index,:])
    else: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            func_values = u_exact(integration_points[jj:end_index,:]) - my_model(integration_points[jj:end_index,:]).detach()
            err += torch.sum(func_values**2 * weights[jj:end_index,:])
    return err**0.5  

def minimize_linear_layer_explicit_assemble(model,target,weights, integration_points,solver="direct"):
    """assemble the linear system and solve it 
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

def OGAL2FittingReLU4D(my_model,target,N_list,num_epochs, M, k =1, linear_solver = "direct",num_batches = 1): 
    
    """ Orthogonal greedy algorithm (0,1)^4 
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

    weights, integration_points = MonteCarlo_Sobol_dDim_weights_points(M ,d = 4) 

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

    err[0]= torch.mean(func_values_sqrd)**0.5
    all_start_time = time.time()
    
    solver = linear_solver
    print("using linear solver: ",solver)
    N = np.prod(N_list) 
    relu_dict_parameters = generate_relu_dict4D(N_list).t() 
#     num_batches = 2**3 #8, divide dictionary elements 
    batch_size = N//num_batches # batch_size * i : batch_size * (i+1), i = 0,..., num_batches - 1 
    
    for i in range(num_epochs): 
  
        print("epoch: ",i+1, end = '\t')
        if num_neuron == 0: 
            func_values = target(integration_points)
        else: 
            with torch.no_grad(): 
                func_values = target(integration_points) - my_model(integration_points)

        start_time = time.time() 

        output = torch.zeros(N,1)
        
        for j in range(0,N,batch_size): 

            end_index = j + batch_size  
            basis_values_batch = (F.relu( torch.matmul(integration_points,relu_dict_parameters[0:4, j:end_index] ) - relu_dict_parameters[4, j:end_index])**k).T # uses broadcasting    
            output[j:end_index,0]  = (torch.abs(torch.matmul(basis_values_batch,func_values))/M)[:,0]
            
        neuron_index = torch.argmax(output.flatten())
        
        print("argmax time taken, ", time.time() - start_time)
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[0:4, neuron_index]) # 
        list_b.append(-relu_dict_parameters[4,neuron_index])
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
    print("time taken: ",time.time() - all_start_time)
    return err, my_model

def select_greedy_neuron_ind(relu_dict_parameters,my_model,target,integration_weights, integration_points,k,activation = 'relu', memory=2**29):
        
        
    num_neuron = my_model.fc2.weight.size(1) if my_model != None else 0
    if num_neuron == 0: 
        func_values = target(integration_points)
    else: 
        M = integration_points.size(0)
        total_size = num_neuron * M 
        num_batch = total_size//memory + 1 # divide according to memory
        batch_size = M//num_batch
        end_ind = 0 

        func_values = torch.zeros(M,1).to(device)

        for j in range(0,M,batch_size): 
            end_ind = j + batch_size
            func_values[j:end_ind,:] = target(integration_points[j:end_ind,:]) - my_model(integration_points[j:end_ind,:]).detach()


    M = integration_points.size(0)
    dim = integration_points.size(1)
    N = relu_dict_parameters.size(1) # number of dictionary elements 
    output = torch.zeros(N,1)
    num_batches = (N*M)//memory + 1 # decide num_batches according to memory 
    batch_size = N//num_batches 
    print("argmax batch num, ", num_batches)
    for j in range(0,N,batch_size): 

        end_index = j + batch_size  
        basis_values_batch = (F.relu( torch.matmul(integration_points,relu_dict_parameters[0:dim, j:end_index] ) - relu_dict_parameters[dim, j:end_index])**k).T # uses broadcasting    
        output[j:end_index,0]  = (torch.abs(torch.matmul(basis_values_batch,func_values))/M)[:,0]

    neuron_index = torch.argmax(output.flatten())
    return neuron_index 

def OGAL2FittingReLU4D_QMC(my_model,target,s,N0,num_epochs, M, k =1, linear_solver = "direct",memory = 2**29): 
    
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
    gw_expand, integration_points = MonteCarlo_Sobol_dDim_weights_points(M ,d = 4) 

    dim = integration_points.size(1)
    M = integration_points.size(0)
    err = torch.zeros(num_epochs+1)    
    num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
    total_size2 = M*(num_neuron+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # in
    if my_model == None: 
        list_b,list_w = [],[]
    else:
        bias = my_model.fc1.bias.detach().data
        nnweights = my_model.fc1.weight.detach().data
        list_b,list_w = list(bias), list(nnweights)

    err[0] = compute_l2_error(target,my_model,M,batch_size_2,gw_expand,integration_points)
    
    all_start_time = time.time()
    solver = linear_solver
    print("using linear solver: ",solver)
    for i in range(num_epochs): 
        print("epoch: ",i+1, end = '\t')
        # relu_dict_parameters = generate_relu_dict4D_QMC(s,N0).t()  
        relu_dict_parameters = generate_relu_dict4plusD_sphere(dim,s,N0).to(device).t() 
    
        neuron_index = select_greedy_neuron_ind(relu_dict_parameters,my_model,target,gw_expand, integration_points, k,activation = 'relu',memory=memory)
        # if num_neuron == 0: 
        #     func_values = target(integration_points)
        # else: 
        #     func_values = target(integration_points) - my_model(integration_points).detach()        
        # N = s*N0 
        # output = torch.zeros(N,1)
        # batch_size = N//num_batches
        
        # for j in range(0,N,batch_size): 

        #     end_index = j + batch_size  
        #     basis_values_batch = (F.relu( torch.matmul(integration_points,relu_dict_parameters[0:4, j:end_index] ) - relu_dict_parameters[4, j:end_index])**k).T # uses broadcasting    
        #     output[j:end_index,0]  = (torch.abs(torch.matmul(basis_values_batch,func_values))/M)[:,0]
        # neuron_index = torch.argmax(output.flatten())

        list_w.append(relu_dict_parameters[0:dim, neuron_index]) # 
        list_b.append(-relu_dict_parameters[dim,neuron_index])
        num_neuron += 1
        my_model = model(dim,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        sol = minimize_linear_layer_explicit_assemble(my_model,target,gw_expand,integration_points, solver)
        print("\t\t time taken minimize linear layer: ",time.time() - start_time) 
        my_model.fc2.weight.data[0,:] = sol[:]

        err[i+1]= compute_l2_error(target,my_model,M,batch_size_2,gw_expand,integration_points)
        print("current error: ",err[i+1]) 
    print("total duration: ",time.time() - all_start_time)
    return err, my_model
