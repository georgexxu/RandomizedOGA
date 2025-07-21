# %%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import sympy as sp 
from scipy.sparse import linalg
from pathlib import Path
import os 
if torch.cuda.is_available():  
    device = "cuda" 
else:  
    device = "cpu" 

pi = torch.tensor(np.pi,dtype=torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
ZERO = torch.tensor([0.]).to(device)

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
    def evaluate_derivative(self, x, i):
        if self.k == 1:
            u1 = self.fc2(torch.heaviside(self.fc1(x),ZERO) * self.fc1.weight.t()[i-1:i,:] )
        else:
            u1 = self.fc2(self.k*F.relu(self.fc1(x))**(self.k-1) *self.fc1.weight.t()[i-1:i,:] )  
        return u1
    def evaluate_2ndderivative(self,x,i,j): 
        if self.k == 2:
            u1 = self.fc2( 2 * torch.heaviside(self.fc1(x),ZERO) * (self.fc1.weight.t()[i-1:i,:])*self.fc1.weight.t()[j-1:j,:]) 
        else:
            u1 = self.fc2( self.k*(self.k-1)*F.relu(self.fc1(x))**(self.k-2) * (self.fc1.weight.t()[i-1:i,:])* (self.fc1.weight.t()[j-1:j,:]))  
        return u1

def plot_2D(f): 
    
    Nx = 400
    Ny = 400 
    xs = np.linspace(0, 1, Nx)
    ys = np.linspace(0, 1, Ny)
    x, y = np.meshgrid(xs, ys, indexing='xy')
    xy_comb = np.stack((x.flatten(),y.flatten())).T
    xy_comb = torch.tensor(xy_comb)
    z = f(xy_comb).reshape(Nx,Ny)
    z = z.detach().numpy()
    plt.figure(dpi=200)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x , y , z )

    plt.show()

def plot_subdomains(my_model):
    x_coord =torch.linspace(0,1,200)
    wi = my_model.fc1.weight.data
    bi = my_model.fc1.bias.data 
    for i, bias in enumerate(bi):  
        if wi[i,1] !=0: 
            plt.plot(x_coord, - wi[i,0]/wi[i,1]*x_coord - bias/wi[i,1])
        else: 
            plt.plot(x_coord,  - bias/wi[i,0]*torch.ones(x_coord.size()))

    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.legend()
    plt.show()
    return 0   

## Initialization
def adjust_neuron_position(my_model,target=None):
    my_model = my_model.cpu()
    counter = 0 
#     positions = torch.tensor([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]])
    positions = torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
    neuron_num = my_model.fc1.bias.size(0)
    for i in range(neuron_num): 
        w = my_model.fc1.weight.data[i:i+1,:]
        b = my_model.fc1.bias.data[i]
        values = torch.matmul(positions,w.T) # + b
        left_end = - torch.max(values)
        right_end = - torch.min(values) 
        off_set = (right_end - left_end)/1000 
        if b <= left_end + off_set: # nearly vanishing
            b = torch.rand(1)*(right_end - left_end - off_set*2) + left_end + off_set 
            my_model.fc1.bias.data[i] = b 
        if b >= right_end - off_set: # nearly nonvanishing everywhere
            if counter < 3:
                counter += 1
            else: # 3 or more 
                b = torch.rand(1)*(right_end - left_end - off_set*2) + left_end + off_set
                my_model.fc1.bias.data[i] = b 
    return my_model

def PiecewiseGQ2D_weights_points(Nx, order): 
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

#     print("order: ",order )
    x, w = np.polynomial.legendre.leggauss(order)
    gauss_pts = np.array(np.meshgrid(x,x,indexing='ij')).reshape(2,-1).T
    weights =  (w*w[:,None]).ravel()

    gauss_pts =torch.tensor(gauss_pts)
    weights = torch.tensor(weights)

    h = 1/Nx # 100 intervals 
    long_weights =  torch.tile(weights,(Nx**2,1))
    long_weights = long_weights.reshape(-1,1)
    long_weights = long_weights * h**2 /4 

    integration_points = torch.tile(gauss_pts,(Nx**2,1))
    scale_factor = h/2 
    integration_points = scale_factor * integration_points

    index = np.arange(1,Nx+1)-0.5
    ordered_pairs = np.array(np.meshgrid(index,index,indexing='ij'))
    ordered_pairs = ordered_pairs.reshape(2,-1).T

    # print(ordered_pairs)
    # print()
    ordered_pairs = torch.tensor(ordered_pairs)
    # print(ordered_pairs.size())
    ordered_pairs = torch.tile(ordered_pairs, (1,order**2)) # number of GQ points
    # print(ordered_pairs)

    ordered_pairs =  ordered_pairs.reshape(-1,2)
    # print(ordered_pairs)
    translation = ordered_pairs*h 
    # print(translation)

    integration_points = integration_points + translation 
#     print(integration_points.size())
    # func_values = integrand2_torch(integration_points)
    return long_weights.to(device), integration_points.to(device)

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

    return long_weights.to(device), integration_points.to(device)

def minimize_linear_layer_H2_explicit_assemble_efficient_general_dim(model,target,weights, integration_points,activation = 'relu',solver="direct",memory = 2**28 ):
    """Biharmonic equation solver
    \Delta^2 u + u = f, in \Omega
    \Delta u = 0, \partial_n (\Delta u) = 0  on \partial \Omega
    """
    # weights, integration_points = PiecewiseGQ2D_weights_points(Nx, order) 
    # integration_points.requires_grad_(True) 
    start_time = time.time() 
    w = model.fc1.weight.data 
    b = model.fc1.bias.data 
    neuron_num = b.size(0) 
    dim = integration_points.size(1) 
    M = integration_points.size(0)

    total_size = neuron_num * M # memory, number of floating numbers 
    print('total size: {} {} = {}'.format(neuron_num,M,total_size))
    num_batch = total_size//memory + 1 # divide according to memory
    print("num batches: ",num_batch)
    batch_size = M//num_batch
    start_ind = 0
    end_ind = 0 
    jac = torch.zeros(b.size(0),b.size(0)).to(device)
    rhs = torch.zeros(b.size(0),1).to(device)
    
    start_time = time.time() 
    for j in range(0,M,batch_size): # batch operation in data points 
        end_ind = j + batch_size
        basis_value_col = F.relu(integration_points[j:end_ind] @ w.t()+ b)**(model.k) 
        weighted_basis_value_col = basis_value_col * weights[j:end_ind] 
        jac += weighted_basis_value_col.t() @ basis_value_col 
        rhs += weighted_basis_value_col.t() @ (target(integration_points[j:end_ind,:])) 

#     if activation == 'relu':
#         assert model.k != 1, "k must not be 1"  
#         basis_value_col = F.relu(integration_points @ w.t()+ b)**(model.k) 
    
    
#     weighted_basis_value_col = basis_value_col * weights 
#     ## assemble the mass matrix term
#     jac = weighted_basis_value_col.t() @ basis_value_col  
#     rhs = weighted_basis_value_col.t() @ (target(integration_points)) 
    print("assembling the mass matrix time taken: ", time.time()-start_time) 
    
    
    ## assemble the biharmonic term 
    for i in range(1,dim + 1): 
        for j in range(i,dim + 1): 
            if i == j: 
                for jj in range(0,M,batch_size):## batch operation 
                    end_ind = jj + batch_size
                    if model.k == 2:  
                        dxx_basis_value_col = 2 * torch.heaviside(integration_points[jj:end_ind]  @ w.t()+ b, ZERO) * (w.t()[i-1:i,:])**2 
                    else: 
                        dxx_basis_value_col = model.k * (model.k -1) * F.relu(integration_points[jj:end_ind] @ w.t()+ b)**(model.k-2) * (w.t()[i-1:i,:])**2 
                    weighted_dxx_basis_value_col = dxx_basis_value_col * weights[jj:end_ind] 
                    jac += weighted_dxx_basis_value_col.t() @ dxx_basis_value_col 
                
            else: 
                for jj in range(0,M,batch_size):## batch operation 
                    end_ind = jj + batch_size
                    if model.k == 2:  
                        dxy_basis_value_col = 2 * torch.heaviside(integration_points[jj:end_ind] @ w.t()+ b, ZERO) * (w.t()[i-1:i,:])* (w.t()[j-1:j,:]) 
                    else: 
                        dxy_basis_value_col = model.k * (model.k -1) * F.relu(integration_points[jj:end_ind] @ w.t()+ b)**(model.k-2) * (w.t()[i-1:i,:])* (w.t()[j-1:j,:])
                    weighted_dxy_basis_value_col = dxy_basis_value_col * weights[jj:end_ind] 
                    jac += 2 * (weighted_dxy_basis_value_col.t() @ dxy_basis_value_col) 

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
import sympy as sp
import numpy as np
import torch

# Define 3D symbolic variables
x_sym, y_sym, z_sym = sp.symbols('x y z')
vars3d = [x_sym, y_sym, z_sym]

# Define your 3D function, e.g., a separable extension of your 2D function
u_expr_3d = (4 * x_sym * (x_sym - 1))**4 * (4 * y_sym * (y_sym - 1))**4 * (4 * z_sym * (z_sym - 1))**4

# Define a helper function for multi-index derivatives
def multi_derivative(expr, vars, orders):
    for var, order in zip(vars, orders):
        expr = sp.diff(expr, var, order)
    return expr

# For the biharmonic operator in 3D, list the terms:
# Pure fourth derivatives
order0_terms = [(0, 0, 0)]
order1_terms = [(1,0,0),(0,1,0),(0,0,1)] 
order2_terms_pure = [(2, 0, 0), (0,2,0), (0, 0, 2)]
order2_terms_mixed = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]  
pure_terms = [(4, 0, 0), (0, 4, 0), (0, 0, 4)]
# Mixed terms (each counted twice)
mixed_terms = [(2, 2, 0), (2, 0, 2), (0, 2, 2)]

all_order_terms = order0_terms + order1_terms + order2_terms_pure + order2_terms_mixed + pure_terms + mixed_terms 
# lambdify the exact solution 
u_exact_sym_func = sp.lambdify((x_sym, y_sym,z_sym), u_expr_3d, modules='numpy')
# Define wrapper functions that accept PyTorch tensors as input and return torch tensors.
def u_exact(x_tensor):
    # Assume x_tensor is a tensor of shape (N, 2) where each row is (x, y)
    x_np = x_tensor.detach().cpu().numpy()
    # Evaluate the symbolic function using the first and second columns
    result_np = u_exact_sym_func(x_np[:, 0], x_np[:, 1],x_np[:, 2]).reshape(-1,1)
    # Convert result to a torch tensor, preserving the device and dtype of the input
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

# Compute and lambdify each 4th derivative term
lambdified_terms = {}
for orders in all_order_terms:
    deriv_expr = multi_derivative(u_expr_3d, vars3d, orders)
    func = sp.lambdify((x_sym, y_sym, z_sym), deriv_expr, modules='numpy')
    lambdified_terms[orders] = func

# Define wrapper functions that convert torch tensors and evaluate these lambdified functions
def evaluate_term(x_tensor, orders):
    # x_tensor assumed to be shape (N, 3)
    x_np = x_tensor.detach().cpu().numpy()
    result_np = lambdified_terms[orders](x_np[:, 0], x_np[:, 1], x_np[:, 2]).reshape(-1, 1)
    return torch.from_numpy(result_np).to(x_tensor.device).type(x_tensor.dtype)

def rhs_3d(x_tensor):
    # Biharmonic operator in 3D:
    result = (evaluate_term(x_tensor, (4, 0, 0)) +
              evaluate_term(x_tensor, (0, 4, 0)) +
              evaluate_term(x_tensor, (0, 0, 4)) +
              2 * evaluate_term(x_tensor, (2, 2, 0)) +
              2 * evaluate_term(x_tensor, (2, 0, 2)) +
              2 * evaluate_term(x_tensor, (0, 2, 2)) +
            evaluate_term(x_tensor, (0, 0, 0)))  # plus any lower-order term if needed
            #    u_exact(x_tensor)) 
    return result

def compute_l2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points):
    ### depend on some non-local variables 
    err = 0  
    d = integration_points.size(1) 
    zero_tuple = tuple([0]*d) 
    if my_model == None: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            err += integration_weights[jj:end_index,:].t()@ ( evaluate_term(integration_points[jj:end_index,:], zero_tuple))**2
    else: 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            err += integration_weights[jj:end_index,:].t()@ ( evaluate_term(integration_points[jj:end_index,:], zero_tuple)  - my_model(integration_points[jj:end_index,:]).detach())**2
    return err**0.5  

def compute_h2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points):
    ### depend on some non-local variables 
    errh2 = torch.zeros(1,1).to(device) 
    if my_model == None: 
        for kk in range(0,M,batch_size_2): 
            end_index = kk + batch_size_2 
            for multi_index in order2_terms_pure: 
                ii = [i+1 for i, value in enumerate(multi_index) if value != 0][0] 
                errh2 += integration_weights[kk:end_index,:].t()@(evaluate_term(integration_points[kk:end_index,:],orders= multi_index))**2
            for multi_index in order2_terms_mixed:
                ii,jj = [i+1 for i, value in enumerate(multi_index) if value != 0] 
                errh2 += 2 * integration_weights[kk:end_index,:].t()@(evaluate_term(integration_points[kk:end_index,:],orders= multi_index))**2
    else: 
        for kk in range(0,M,batch_size_2): 
            end_index = kk + batch_size_2 
            for multi_index in order2_terms_pure: 
                ii = [i+1 for i, value in enumerate(multi_index) if value != 0][0] 
                errh2 += integration_weights[kk:end_index,:].t()@(evaluate_term(integration_points[kk:end_index,:],orders= multi_index) - my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,ii).detach())**2
            for multi_index in order2_terms_mixed:
                ii,jj = [i+1 for i, value in enumerate(multi_index) if value != 0] 
                errh2 += 2 * integration_weights[kk:end_index,:].t()@(evaluate_term(integration_points[kk:end_index,:],orders= multi_index) - my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,jj).detach())**2

    return errh2**0.5  

# %%
def OGABiharmonicReLU3D(my_model,target, N_list,num_epochs, Nx, order, k =1, rand_deter = 'rand', linear_solver = "direct",memory = 2**28): 

    integration_weights, integration_points = PiecewiseGQ3D_weights_points(Nx, order)

    err = torch.zeros(num_epochs+1)
    errh2 = torch.zeros(num_epochs+1)
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
    M = integration_points.size(0) 
    dim = integration_points.size(1)  

    num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
    total_size2 = M*(num_neuron+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # in

    err[0]= compute_l2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points)
    errh2[0] = compute_h2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points)
    start_time = time.time()
    solver = linear_solver

    N0 = np.prod(N_list)

    for i in range(num_epochs): 
        print("epoch: ",i+1, end = '\t')
        if rand_deter == 'rand':
            relu_dict_parameters = generate_relu_dict4plusD_sphere(dim,1,N0).to(device)  

        neuron_num = my_model.fc2.weight.size(1) if my_model != None else 0
        total_size2 = M*(neuron_num+1)
        num_batch2 = total_size2//memory + 1 
        batch_size_2 = M//num_batch2 # integration points 
        print("num batches argmax: ",num_batch2)
        func_values = torch.zeros(M,1).to(device)  

        if num_neuron == 0: 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2
                func_values[jj:end_index] += - target(integration_points[jj:end_index])
        else: 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2
                func_values[jj:end_index] += - target(integration_points[jj:end_index])
                func_values[jj:end_index] += my_model(integration_points[jj:end_index]).detach()

        weight_func_values = func_values*integration_weights  
        total_size = M * N0 
        num_batch = total_size//memory + 1 
        batch_size_1 = N0//num_batch # dictionary elements
        output1 = torch.zeros(N0,1).to(device) 
        print("num batches argmax: ",num_batch)
        for jj in range(0,N0,batch_size_1):  
            end_index = jj + batch_size_1 
            basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[jj:end_index,0:dim].T ) - relu_dict_parameters[jj:end_index,dim])**k) # uses broadcasting
            output1[jj:end_index] += basis_values.t()@weight_func_values #

        output2 = torch.zeros(output1.size()).to(device) 
        assert k != 1, "k must not be 1"  
        for ii in range(1,dim+1):
            for jj in range(ii,dim+1):
                for kk in range(0,M,batch_size_2): 
                    end_index = kk + batch_size_2 
                    if ii == jj: 
                        if k == 2:  
                            dxx_basis_values = 2 * torch.heaviside(integration_points[kk:end_index,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim], ZERO) * (relu_dict_parameters.t()[ii-1:ii,:])**2  
                            if my_model!= None:
                                dxx_my_model = my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,ii).detach()
                                output2 += dxx_basis_values.t() @ (dxx_my_model*integration_weights[kk:end_index,:]) 
                        else:  
                            dxx_basis_values = k *(k-1) * F.relu(integration_points[kk:end_index,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim])**(k-2) * (relu_dict_parameters.t()[ii-1:ii,:])**2 

                            if my_model!= None:
                                dxx_my_model = my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,ii).detach()

                                output2 += dxx_basis_values.t() @ (dxx_my_model*integration_weights[kk:end_index,:]) 
                    else:    
                        if k == 2:  
                            dxy_basis_values = 2 * torch.heaviside(integration_points[kk:end_index,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim], ZERO) * (relu_dict_parameters.t()[ii-1:ii,:]) * (relu_dict_parameters.t()[jj-1:jj,:]) 
                            if my_model!= None:
                                dxy_my_model = my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,jj).detach() 
                                output2 += 2 * dxy_basis_values.t() @ (dxy_my_model*integration_weights[kk:end_index,:])  
                        else:  
                            dxy_basis_values = k *(k-1) * F.relu(integration_points[kk:end_index,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim])**(k-2) * (relu_dict_parameters.t()[ii-1:ii,:]) * (relu_dict_parameters.t()[jj-1:jj,:])
                            if my_model!= None:
                                dxy_my_model = my_model.evaluate_2ndderivative(integration_points[kk:end_index,:],ii,jj).detach() 
                                output2 += 2 *  dxy_basis_values.t() @ (dxy_my_model*integration_weights[kk:end_index,:])  

        if my_model!= None:
            output = torch.abs(output1 + output2) 
        else: 
            output = torch.abs(output1) 
        # output = torch.abs(torch.matmul(basis_values,weight_func_values)) # 
        neuron_index = torch.argmax(output.flatten())
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:dim]) # 
        list_b.append(-relu_dict_parameters[neuron_index,dim])
        num_neuron += 1
        my_model = model(dim,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        #Todo Done 
        # sol = minimize_linear_layer_H2_explicit_assemble_efficient(my_model,target,gw_expand, integration_points,activation = 'relu',solver = solver)
        sol = minimize_linear_layer_H2_explicit_assemble_efficient_general_dim(my_model,target,integration_weights, integration_points,activation = 'relu',solver = solver,memory = memory)
        my_model.fc2.weight.data[0,:] = sol[:]

        # L2 error ||u - u_n|| and H2 error 
        num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
        total_size2 = M*(num_neuron+1)
        num_batch2 = total_size2//memory + 1 
        batch_size_2 = M//num_batch2 # in
        err[i+1]= compute_l2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points)
        errh2[i+1] = compute_h2_error_biharmonic(my_model,M,batch_size_2,integration_weights,integration_points)

    print("time taken: ",time.time() - start_time)
    return err, errh2, my_model

def generate_relu_dict4plusD_sphere(dim, s,N0): # 
    samples = torch.randn(s*N0,dim +1) 
    samples = samples/samples.norm(dim=1,keepdim=True)  
    Wb = samples 
    return Wb 

# %%
def show_convergence_order(err_l2,err_h10,exponent,dict_size, filename,write2file = False):
    
    if write2file:
        file_mode = "a" if os.path.exists(filename) else "w"
        f_write = open(filename, file_mode)
    
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    err_list2 = [err_h10[i] for i in neuron_nums ] 
    # f_write.write('M:{}, relu {} \n'.format(M,k))
    if write2file:
        f_write.write('dictionary size: {}\n'.format(dict_size))
        f_write.write("neuron num \t\t error \t\t order \t\t h10 error \\ order \n")
    print("neuron num \t\t error \t\t order")
    for i, item in enumerate(err_list):
        if i == 0: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t")
            
            # print("*")
            print("{} \t\t {:.6f} \t\t * \t\t {:.6f} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t * \t\t {} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ))
        else: 
            # print(neuron_nums[i], end = "\t\t")
            # print(item, end = "\t\t") 
            # print(np.log(err_list[i-1]/err_list[i])/np.log(2))
            print("{} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t {} \t\t {} \t\t {} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ))
    if write2file:     
        f_write.write("\n")
        f_write.close()

# %%

function_name = "biharmonic" 
filename_write = "3DOGA-{}-order.txt".format(function_name)
f_write = open(filename_write, "a")
f_write.write("\n")
f_write.close() 
save = True  
relu_k= 3 
memory = 2**28
trial_num = 5
for trial in range(trial_num): 
    for N_list in [[2**3,2**3,2**4]]: #[2**6,2**6],[2**7,2**7] 
        # save = True 
        f_write = open(filename_write, "a")
        my_model = None 
        Nx = 50
        order = 2   
        exponent = 10
        num_epochs = 2**exponent  
        N = np.prod(N_list)
        errl2,errh2, my_model = OGABiharmonicReLU3D(my_model,rhs_3d, N_list,num_epochs, Nx, order, k = relu_k, rand_deter= 'rand', linear_solver = "direct",memory = memory)
        if save: 
            folder = 'data-biharmonic/'
            filename = folder + 'errl2_OGA_3D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(errl2,filename)
            filename = folder + 'errh2_OGA_3D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(errh2,filename)
            filename = folder + 'model_OGA_3D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(my_model.state_dict(),filename)

    show_convergence_order(errl2,errh2,exponent,N, filename_write,write2file = True)


# %%



