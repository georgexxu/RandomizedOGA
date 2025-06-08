# %%
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
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 

torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi,dtype=torch.float64)

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
#     err_list2 = [err_h10[i] for i in neuron_nums ] 
    # f_write.write('M:{}, relu {} \n'.format(M,k))
    # f_write.write('randomized dictionary size: {}\n'.format(N))
    # f_write.write("neuron num \t\t error \t\t order \t\t h10 error \\ order \n")
    l2_order = -1/2-(2*k + 1)/(2*d)
#     h10_order = -1/2-(2*(k-1) + 1)/(2*d)
#     print("neuron num  & \t $\|u-u_n \|_{L^2}$ & \t order $O(n^{{{}})$ & \t $ | u -u_n |_{H^1}$ & \t order $O(n^{{{}})$ \\\ \hline \hline ".format(l2_order,h10_order))
    print("neuron num  & \t $\\|u-u_n \\|_{{L^2}}$ & \t order $O(n^{{{:.2f}}})$  \\\\ \\hline \\hline ".format(l2_order))
    for i, item in enumerate(err_list):
        if i == 0: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t")

            # print("*")
            print("{} \t\t & {:.6f} &\t\t *  \\\ \hline  \n".format(neuron_nums[i],item) )   
            # f_write.write("{} \t\t {} \t\t * \t\t {} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ))
        else: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t") 
            # print(np.log(err_list[i-1]/err_list[i])/np.log(2))
            print("{} \t\t &  {:.3e} &  \t\t {:.2f} \\\ \hline  \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2) ) )
            # f_write.write("{} \t\t {} \t\t {} \t\t {} \t\t {} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ))
    # f_write.write("\n")
    # f_write.close()



# %%
def PiecewiseGQ3D_weights_points(Nx, order): 
    """ A slight modification of PiecewiseGQ2D function that only needs the weights and integration points.
    Parameters
    ----------

    Nx: int 
        number of intervals along the dimension. No Ny, assume Nx = Ny
    order: int 
        order of the Gauss Quadrature

    Returns
    -------
    long_weights: torch.tensor
    integration_points: torch.tensor
    """

    """
    Parameters
    ----------
    target : 
        Target function 
    Nx: int 
        number of intervals along the dimension. No Ny, assume Nx = Ny
    order: int 
        order of the Gauss Quadrature
    """

    # print("order: ",order )
    x, w = np.polynomial.legendre.leggauss(order)
    gauss_pts = np.array(np.meshgrid(x,x,x,indexing='ij')).reshape(3,-1).T
    weight_list = np.array(np.meshgrid(w,w,w,indexing='ij'))
    weights =   (weight_list[0]*weight_list[1]*weight_list[2]).ravel() 

    gauss_pts =torch.tensor(gauss_pts)
    weights = torch.tensor(weights)

    h = 1/Nx # 100 intervals 
    long_weights =  torch.tile(weights,(Nx**3,1))
    long_weights = long_weights.reshape(-1,1)
    long_weights = long_weights * h**3 /8 

    integration_points = torch.tile(gauss_pts,(Nx**3,1))
    # print("shape of integration_points", integration_points.size())
    scale_factor = h/2 
    integration_points = scale_factor * integration_points

    index = np.arange(1,Nx+1)-0.5
    ordered_pairs = np.array(np.meshgrid(index,index,index,indexing='ij'))
    ordered_pairs = ordered_pairs.reshape(3,-1).T

    # print(ordered_pairs)
    # print()
    ordered_pairs = torch.tensor(ordered_pairs)
    # print(ordered_pairs.size())
    ordered_pairs = torch.tile(ordered_pairs, (1,order**3)) # number of GQ points
    # print(ordered_pairs)

    ordered_pairs =  ordered_pairs.reshape(-1,3)
    # print(ordered_pairs)
    translation = ordered_pairs*h 
    # print(translation)

    integration_points = integration_points + translation 

    return long_weights, integration_points

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


# %%


# %% [markdown]
# ## 3D QMC greedy algorithm 
# 

# %%
def generate_relu_dict3D(N_list):
    N1 = N_list[0]
    N2 = N_list[1]
    N3 = N_list[2]
    
    N = N1*N2*N3 
    theta1 = np.linspace(0, pi, N1, endpoint= True).reshape(N1,1)
    theta2 = np.linspace(0, 2*pi, N2, endpoint= False).reshape(N2,1)
    b = np.linspace(-1.732, 1.732, N3,endpoint=False).reshape(N3,1) # threshold: 3**0.5  
    coord3 = np.array(np.meshgrid(theta1,theta2,b,indexing='ij'))
    coord3 = coord3.reshape(3,-1).T # N1*N2*N3 x 3. coordinates for the grid points 
    coord3 = torch.tensor(coord3) 

    f1 = torch.zeros(N,1) 
    f2 = torch.zeros(N,1)
    f3 = torch.zeros(N,1)
    f4 = torch.zeros(N,1)

    f1[:,0] = torch.cos(coord3[:,0]) 
    f2[:,0] = torch.sin(coord3[:,0]) * torch.cos(coord3[:,1])
    f3[:,0] = torch.sin(coord3[:,0]) * torch.sin(coord3[:,1])
    f4[:,0] = coord3[:,2] 

    Wb_tensor = torch.cat([f1,f2,f3,f4],1) # N x 4 
    return Wb_tensor

def generate_relu_dict3D_QMC(s,N0):
    #Sob = torch.quasirandom.SobolEngine(dimension =3, scramble= True, seed=None) 
    #samples = Sob.draw(N0).double() 

    # Monte Carlo 
    samples = torch.rand(s*N0,3) 
    T =torch.tensor([[pi,0,0],[0,2*pi,0],[0,0,1.732*2]])
    shift = torch.tensor([0,0,-1.732])
    samples = samples@T + shift 

    f1 = torch.zeros(s*N0,1) 
    f2 = torch.zeros(s*N0,1)
    f3 = torch.zeros(s*N0,1)
    f4 = torch.zeros(s*N0,1)

    f1[:,0] = torch.cos(samples[:,0]) 
    f2[:,0] = torch.sin(samples[:,0]) * torch.cos(samples[:,1])
    f3[:,0] = torch.sin(samples[:,0]) * torch.sin(samples[:,1])
    f4[:,0] = samples[:,2] 

    Wb_tensor = torch.cat([f1,f2,f3,f4],1) # N x 4 
    return Wb_tensor

def MonteCarlo_Sobol_dDim_weights_points(M ,d = 4):
    Sob_integral = torch.quasirandom.SobolEngine(dimension =d, scramble= False, seed=None) 
    integration_points = Sob_integral.draw(M).double() 
    integration_points = integration_points.to(device)
    weights = torch.ones(M,1).to(device)/M 
    return weights.to(device), integration_points.to(device) 

def generate_relu_dict4plusD_sphere(dim, s,N0): # 
    samples = torch.randn(s*N0,dim +1) 
    samples = samples/samples.norm(dim=1,keepdim=True)  
    Wb = samples 
    return Wb 



# %%
def OGAL2FittingReLU3D(my_model,target,N_list,num_epochs,plot_freq, Nx, order, k =1, linear_solver = "direct"): 

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
    gw_expand, integration_points = PiecewiseGQ3D_weights_points(Nx, order)
    gw_expand = gw_expand.to(device)
    integration_points = integration_points.to(device)

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

    err[0]= torch.sum(func_values_sqrd*gw_expand)**0.5
    start_time = time.time()
    
    solver = linear_solver
    print("using linear solver: ",solver)
#     relu_dict_parameters = generate_relu_dict3D_QMC(s,N0).to(device) 
    N = np.prod(N_list) 
    relu_dict_parameters = generate_relu_dict3D(N_list).to(device)
    
    for i in range(num_epochs): 
#         relu_dict_parameters = generate_relu_dict3D_QMC(s,N0).to(device) 
        print("epoch: ",i+1, end = '\t')
        if num_neuron == 0: 
            func_values = target(integration_points)
        else: 
            func_values = target(integration_points) - my_model(integration_points).detach()

        weight_func_values = func_values*gw_expand  
        basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[:,0:3].T ) - relu_dict_parameters[:,3])**k).T # uses broadcasting

        output = torch.abs(torch.matmul(basis_values,weight_func_values)) # 
        neuron_index = torch.argmax(output.flatten())
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:3]) # 
        list_b.append(-relu_dict_parameters[neuron_index,3])
        num_neuron += 1
        my_model = model(3,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

#         sol = minimize_linear_layer(my_model,target,solver,Nx,order)
        sol = minimize_linear_layer_explicit_assemble(my_model,target,gw_expand,integration_points,solver)

        my_model.fc2.weight.data[0,:] = sol[:]
        # if (i+1)%plot_freq == 0: 
        #     plot_2D(my_model.cpu())
        #     my_model = my_model.to(device)

        func_values = target(integration_points) - my_model(integration_points).detach()
        func_values_sqrd = func_values*func_values

        #Todo Done 
        err[i+1]= torch.sum(func_values_sqrd*gw_expand)**0.5
    print("time taken: ",time.time() - start_time)
    return err, my_model


def select_greedy_neuron_ind(relu_dict_parameters,my_model,target,integration_weights, integration_points, k,activation = 'relu',memory=2**29): 

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
    N = relu_dict_parameters.size(0) # number of neurons in the dictionary 
    dim = integration_points.size(1) # dimension of the integration points
    output = torch.zeros(N,1)
    num_batches = (N*M)//memory + 1 # decide num_batches according to memory 
    batch_size = N//num_batches 
    print("argmax batch num, ", num_batches)
    weight_func_values = func_values*integration_weights 
    for j in range(0,N,batch_size): 
        end_index = j + batch_size 
        basis_values_batch = (F.relu( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])**k).T # uses broadcasting
        output[j:end_index,0]  = torch.abs(basis_values_batch @ weight_func_values)[:,0]

    neuron_index = torch.argmax(output.flatten())

    return neuron_index


def OGAL2FittingReLU3D_QMC(my_model,target,s,N0,num_epochs,plot_freq, Nx, order, k =1, linear_solver = "direct",memory = 2**29): 

    """ Orthogonal greedy algorithm on (0,1)^3 
    """
    #Todo Done
    gw_expand, integration_points = PiecewiseGQ3D_weights_points(Nx, order)
    gw_expand = gw_expand.to(device)
    integration_points = integration_points.to(device)
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
    
    solver = linear_solver
    print("using linear solver: ",solver)
    start_time = time.time()
    for i in range(num_epochs): 
        print("epoch: ",i+1, end = '\t')
        # relu_dict_parameters = generate_relu_dict3D_QMC(s,N0).to(device) 
        relu_dict_parameters = generate_relu_dict4plusD_sphere(dim,s,N0).to(device)
        
        neuron_index = select_greedy_neuron_ind(relu_dict_parameters,my_model,target,gw_expand, integration_points, k,activation = 'relu',memory=memory)  
        
        # if num_neuron == 0: 
        #     func_values = target(integration_points)
        # else: 
        #     func_values = target(integration_points) - my_model(integration_points).detach()

        # weight_func_values = func_values*gw_expand  
        # basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[:,0:3].T ) - relu_dict_parameters[:,3])**k).T # uses broadcasting

        # output = torch.abs(torch.matmul(basis_values,weight_func_values)) # 
        # neuron_index = torch.argmax(output.flatten())
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:dim]) # 
        list_b.append(-relu_dict_parameters[neuron_index,dim]) # Changed from 3 to dim])
        num_neuron += 1
        my_model = model(dim,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        sol = minimize_linear_layer_explicit_assemble(my_model,target,gw_expand,integration_points,solver)
        my_model.fc2.weight.data[0,:] = sol[:]

        err[i+1] = compute_l2_error(target,my_model,M,batch_size_2,gw_expand,integration_points)
    print("time taken: ",time.time() - start_time)
    return err, my_model


