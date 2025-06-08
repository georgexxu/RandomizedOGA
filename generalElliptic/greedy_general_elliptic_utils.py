# %%
## 1D General elliptic PDE of the following form: 
## -div( a(x) grad u(x)) + b(x) grad u(x) + c(x) u(x) = f(x) in [0,1] 
## a(x), b(x), c(x) are set to be constant functions 
## du_dn = g on the boundary 
## this version also contains using the tanh-activated shallow neural network to solve the PDE 
"""
log
Nov 17th 2024 Modified by Xiaofeng: 
added three functions   
1. select_discrete_dictionary
2. compute_l2_error 
3. compute_gradient_error

Nov 20th 2024 Modified by Xiaofeng 
1. use an efficient way to assemble the matrix that reuses previous matrices 
    - minimize_linear_layer_efficient

Todo: 
1. remove some redundant variable to save memory 
2. test a huge dictionary and huge quadrature points  
"""
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
pi = torch.tensor(np.pi)
ZERO = torch.tensor([0.]).to(device)

###===============model parameters below================================
LAMBDA = -4 # c(x) = LAMBDA, if negative Helmholtz equation parameters
BETA = 5 ## convection term parameters 
DIMENSION = 3  ## dimension of the problem 
###===============model parameters above================================

## Define the neural network model
## already general in any dimension
class model_tanh(nn.Module):
    """ cosine shallow neural network
    Parameters: 
    input size: input dimension
    hidden_size1 : number of hidden layers 
    num_classes: output classes 
    """
    def __init__(self, input_size, hidden_size1, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, num_classes,bias = False)
    def forward(self, x):
        u1 = self.fc2( torch.tanh(self.fc1(x)) )
        return u1
    
    def tanh_activation_dx(self,x): 
        return 1/torch.cosh(x)**2  
      
    def evaluate_derivative(self, x, i):
        u1 = self.fc2( self.tanh_activation_dx(self.fc1(x)) *self.fc1.weight.t()[i-1:i,:] )  
        return u1

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
            ## ZERO = torch.tensor([0.]).to(device)
            u1 = self.fc2(torch.heaviside(self.fc1(x),ZERO) * self.fc1.weight.t()[i-1:i,:] )
        else:
            u1 = self.fc2(self.k*F.relu(self.fc1(x))**(self.k-1) *self.fc1.weight.t()[i-1:i,:] )  
        return u1

# %%
def plot_solution_modified(r1,r2,model,x_test,u_true,name=None): 
    # Plot function: test results 
    u_model_cpu = model(x_test).cpu().detach()
    
    w = model.fc1.weight.data.squeeze()
    b = model.fc1.bias.data.squeeze()
    x_model_pt = (-b/w).view(-1,1)
    x_model_pt = x_model_pt[x_model_pt>=r1].reshape(-1,1)
    u_model_pt = model(x_model_pt).cpu().detach()
    plt.figure(dpi = 100)
    plt.plot(x_test.cpu(),u_model_cpu,'-.',label = "nn function")
    plt.plot(x_test.cpu(),u_true.cpu(),label = "true")
    # plt.plot(x_model_pt.cpu(),u_model_pt.cpu(),'.r')
    if name!=None: 
        plt.title(name)
    plt.legend()
    plt.show()

# %%
def PiecewiseGQ1D_weights_points(x_l,x_r,Nx, order):
    """ Output the coeffients and weights for piecewise Gauss Quadrature 
    Parameters
    ----------
    x_l : float 
    left endpoint of an interval 
    x_r: float
    right endpoint of an interval 
    Nx: int 
    number of subintervals for integration
    order: int
    order of Gauss Quadrature 
    Returns
    -------
    vectorized quadrature weights and integration points
    """
    x,w = np.polynomial.legendre.leggauss(order)
    gx = torch.tensor(x).to(device)
    gx = gx.view(1,-1) # row vector 
    gw = torch.tensor(w).to(device)    
    gw = gw.view(-1,1) # Column vector 
    nodes = torch.linspace(x_l,x_r,Nx+1).view(-1,1).to(device) 
    coef1 = ((nodes[1:,:] - nodes[:-1,:])/2) # n by 1  
    coef2 = ((nodes[1:,:] + nodes[:-1,:])/2) # n by 1  
    coef2_expand = coef2.expand(-1,gx.size(1)) # Expand to n by p shape, -1: keep the first dimension n , expand the 2nd dim (columns)
    integration_points = coef1@gx + coef2_expand
    integration_points = integration_points.flatten().view(-1,1) # Make it a column vector
    gw_expand = torch.tile(gw,(Nx,1)) # rows: n copies of current tensor, columns: 1 copy, no change
    # Modify coef1 to be compatible with func_values
    coef1_expand = coef1.expand(coef1.size(0),gx.size(1))    
    coef1_expand = coef1_expand.flatten().view(-1,1)
    return coef1_expand.to(device) * gw_expand.to(device), integration_points.to(device)

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

def MonteCarlo_Sobol_dDim_weights_points(M ,d = 4):
    Sob_integral = torch.quasirandom.SobolEngine(dimension =d, scramble= False, seed=None) 
    integration_points = Sob_integral.draw(M).double() 
    integration_points = integration_points.to(device)
    weights = torch.ones(M,1).to(device)/M 
    return weights, integration_points 
def Neumann_boundary_quadrature_points_weights(M,d):
    def generate_quadpts_on_boundary(gw_expand_bd, integration_points_bd,d):
        size_pts_bd = integration_points_bd.size(0) 
        gw_expand_bd_faces = torch.tile(gw_expand_bd,(2*d,1)) # 2d boundaries, 拉成长条

        integration_points_bd_faces = torch.zeros(2*d*integration_points_bd.size(0),d).to(device)
        for ind in range(d): 
            integration_points_bd_faces[2 *ind * size_pts_bd :(2 *ind +1) * size_pts_bd,ind:ind+1] = 0 
            integration_points_bd_faces[(2 *ind)*size_pts_bd :(2 * ind +1) * size_pts_bd,:ind] = integration_points_bd[:,:ind]
            integration_points_bd_faces[(2 *ind)*size_pts_bd :(2 * ind +1) * size_pts_bd,ind+1:] = integration_points_bd[:,ind:]

            integration_points_bd_faces[(2 *ind +1) * size_pts_bd:(2 *ind +2)*size_pts_bd,ind:ind+1] = 1
            integration_points_bd_faces[(2 *ind +1) * size_pts_bd:(2 *ind +2)*size_pts_bd,:ind] = integration_points_bd[:,:ind]        
            integration_points_bd_faces[(2 *ind +1) * size_pts_bd:(2 *ind +2)*size_pts_bd,ind+1:] = integration_points_bd[:,ind:]
        return gw_expand_bd_faces, integration_points_bd_faces
    
    if d == 1: 
        print('dim',d)
        gw_expand_bd_faces = torch.tensor([1.,1.]).view(-1,1).to(device)
        integration_points_bd_faces = torch.tensor([0.,1.]).view(-1,1).to(device) 
    elif d == 2: 
        print('dim',d)
        gw_expand_bd, integration_points_bd = PiecewiseGQ1D_weights_points(0,1,8192, order = 3) 
    elif d == 3: 
        gw_expand_bd, integration_points_bd = PiecewiseGQ2D_weights_points(200, order = 3) 
    elif d == 4: 
        gw_expand_bd, integration_points_bd = PiecewiseGQ3D_weights_points(25, order = 3) 
        print('dim',d)
    else: 
        gw_expand_bd, integration_points_bd = MonteCarlo_Sobol_dDim_weights_points(M ,d = d-1)
        print('dim >=5 ')
    if d != 1: 
        gw_expand_bd_faces, integration_points_bd_faces = generate_quadpts_on_boundary(gw_expand_bd, integration_points_bd,d)
    return gw_expand_bd_faces.to(device), integration_points_bd_faces.to(device) 

def generate_relu_dict3D(N_list):
    N1 = N_list[0]
    N2 = N_list[1]
    N3 = N_list[2]
    
    N = N1*N2*N3 
    theta1 = np.linspace(0, pi, N1, endpoint= True).reshape(N1,1)
    theta2 = np.linspace(0, 2*pi, N2, endpoint= False).reshape(N2,1)
    b = np.linspace(-3**0.5, 3**0.5, N3,endpoint=False).reshape(N3,1) # threshold: 3**0.5  
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
    # Monte Carlo 
    samples = torch.rand(s*N0,3) 
    T =torch.tensor([[pi,0,0],[0,2*pi,0],[0,0,3**0.5 *2]])
    shift = torch.tensor([0,0,-3**0.5])
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

def generate_tanh_dict3D_QMC(s,N0,Rm):
    # Monte Carlo 
    samples = torch.randn(s*N0,4)

    T =torch.tensor([[2*Rm,0.,0.,0.],[0.,2*Rm,0.,0.],[0.,0.,2*Rm,0.],[0.,0.,0.,2*Rm]])
    shift = torch.tensor([-Rm,-Rm,-Rm,-Rm],dtype = torch.float64) 
    samples = samples@T  + shift 

    return samples 

def generate_relu_dict4plusD_sphere(dim, s,N0): # 
    samples = torch.randn(s*N0,dim +1) 
    samples = samples/samples.norm(dim=1,keepdim=True)  
    Wb = samples 
    return Wb 

def generate_tanh_dict3D_QMC_normal(s,N0,var):
    # Monte Carlo 
    samples = torch.normal(0,var,(4,s*N0))

    return samples 

def show_convergence_order2(err_l2,err_h10,exponent,dict_size,k,d, filename,write2file = False):
    
    if write2file:
        file_mode = "a" if os.path.exists(filename) else "w"
        f_write = open(filename, file_mode)
    
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    err_list2 = [err_h10[i] for i in neuron_nums ] 
    l2_order = -1/2-(2*k + 1)/(2*d)
    h1_order =  -1/2-(2*(k-1)+ 1)/(2*d)
    if write2file:
        f_write.write('dictionary size: {}\n'.format(dict_size))
        f_write.write("neuron num \t\t error \t\t order {:.2f} \t\t h10 error \t\t order {:.2f} \n".format(l2_order,h1_order))
    print("neuron num \t\t error \t\t order {:.2f} \t\t h10 error \t\t order {:.2f} \n".format(l2_order,h1_order))
    for i, item in enumerate(err_list):
        if i == 0: 
            print("{} \t\t {:.6f} \t\t * \t\t {:.6f} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t * \t\t {} \t\t * \n".format(neuron_nums[i],item, err_list2[i] ))
        else: 
            print("{} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \t\t {:.6f} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ) )
            if write2file: 
                f_write.write("{} \t\t {} \t\t {} \t\t {} \t\t {} \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ))
    if write2file:     
        f_write.write("\n")
        f_write.close()


def show_convergence_order_latex2(err_l2,err_h10,exponent,k=1,d=1): 
    neuron_nums = [2**j for j in range(2,exponent+1)]
    err_list = [err_l2[i] for i in neuron_nums ]
    err_list2 = [err_h10[i] for i in neuron_nums ] 
    l2_order = -1/2-(2*k + 1)/(2*d)
    h1_order =  -1/2-(2*(k-1)+ 1)/(2*d)
    print("neuron num  & \t $\|u-u_n \|_{{L^2}}$ & \t order $O(n^{{{:.2f}}})$  & \t $ | u -u_n |_{{H^1}}$ & \t order $O(n^{{{:.2f}}})$  \\\ \hline \hline ".format(l2_order,h1_order))
    for i, item in enumerate(err_list):
        if i == 0: 
            print("{} \t\t & {:.6f} &\t\t * & \t\t {:.6f} & \t\t *  \\\ \hline  \n".format(neuron_nums[i],item, err_list2[i] ) )   
        else: 
            print("{} \t\t &  {:.3e} &  \t\t {:.2f} &  \t\t {:.3e} &  \t\t {:.2f} \\\ \hline  \n".format(neuron_nums[i],item,np.log(err_list[i-1]/err_list[i])/np.log(2),err_list2[i] , np.log(err_list2[i-1]/err_list2[i])/np.log(2) ) )


def select_discrete_dictionary(activation,rand_deter,N0,R):
    if isinstance(N0, list):
        N0_num = np.prod(N0)
    else: 
        N0_num = N0 

    if rand_deter == 'deter': 
        if activation == 'relu':
            dict_parameters = generate_relu_dict3D(N0).to(device) 
        elif activation == 'tanh':
            print("for tanh, automatically use randomized dictionary")
            dict_parameters = generate_tanh_dict3D_QMC(1,N0_num,R).to(device)
            
    if rand_deter == 'rand': 
        if activation == 'relu':
            # dict_parameters = generate_relu_dict3D_QMC(1,N0_num).to(device)   
            dict_parameters = generate_relu_dict4plusD_sphere(dim = 3, s = 1,N0 = N0_num).to(device) 
        elif activation == 'tanh':
            dict_parameters = generate_tanh_dict3D_QMC(1,N0_num,R).to(device)
    return dict_parameters 

## helper functions 
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

def compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,weights,integration_points):
    """
    Parameters
    ----------
    u_exact_grad: list or None
        a list that contains ways of evaluating partial derivatives that gives the gradient  
    """
    err_h10 = 0 
     # initial gradient error 
    if u_exact_grad != None and my_model!=None:
        u_grad = u_exact_grad() 
        for ii, grad_i in enumerate(u_grad): 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2 
                my_model_dxi = my_model.evaluate_derivative(integration_points[jj:end_index,:],ii+1).detach() 
                err_h10 += torch.sum((grad_i(integration_points[jj:end_index,:]) - my_model_dxi)**2 * weights[jj:end_index,:])
    elif u_exact_grad != None and my_model==None:
        u_grad = u_exact_grad() 
        for grad_i in u_grad: 
            for jj in range(0,M,batch_size_2): 
                end_index = jj + batch_size_2 
                err_h10 += torch.sum((grad_i(integration_points[jj:end_index,:]))**2 * weights[jj:end_index,:])
    return err_h10**0.5


# %%
def minimize_linear_layer_efficient(Mat, rhs_vec, model,target,weights, integration_points,weights_bd, integration_points_bd, g_N, activation = 'relu', solver = 'direct',memory=2**29):  
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
    neuron_num = b.size(0) 
    M = integration_points.size(0)

    total_size = neuron_num * M # memory, number of floating numbers 
    print('total size: {} {} = {}'.format(neuron_num,M,total_size))
    num_batch = total_size//memory + 1 # divide according to memory
    print("num batches: ",num_batch)
    batch_size = M//num_batch

    coef_func = LAMBDA # 3 * model(integration_points).detach()**2 #changing after each newton iteration 
    # jac = torch.zeros(neuron_num,neuron_num).to(device)
    # rhs = torch.zeros(neuron_num,1).to(device)
    col_k_sym = torch.zeros(neuron_num,1).to(device)
    col_k_nonsym = torch.zeros(neuron_num,1).to(device) # nonsymmetric part from convection term 
    row_k_nonsym = torch.zeros(neuron_num,1).to(device) # nonsymmetric part from convection term 
    rhs_k = torch.zeros(1,1).to(device) 
    for j in range(0,M,batch_size): 
        end_index = j + batch_size
        if activation == 'relu':
            basis_value_col = F.relu(integration_points[j:end_index] @ (w.t())+ b)**(model.k) 
        if activation == 'tanh':
            basis_value_col = torch.tanh(integration_points[j:end_index] @ w.t()+ b)
        weighted_basis_value_col = basis_value_col * weights[j:end_index] 
        
        if activation == 'relu' and model.k == 1:  
            derivative_comm_part = torch.heaviside(integration_points[j:end_index] @ w.t()+ b, ZERO) 
        elif activation == 'relu' and model.k > 1: 
            derivative_comm_part = model.k * F.relu(integration_points[j:end_index] @ w.t()+ b)**(model.k-1)
        elif activation == 'tanh':
            derivative_comm_part = torch.cosh(integration_points[j:end_index] @ w.t()+ b)**(-2)  
            
        # jac += weighted_basis_value_col.t() @ (coef_func * basis_value_col) # mass matrix 
        # rhs += weighted_basis_value_col.t() @ (target(integration_points[j:end_index]) ) #rhs 
        col_k_sym += weighted_basis_value_col.t() @ (coef_func * basis_value_col[:,neuron_num-1:neuron_num])
        rhs_k += (weighted_basis_value_col[:,neuron_num-1:neuron_num]).t() @ (target(integration_points[j:end_index])) #rhs 

        for d in range(DIMENSION): 
            basis_value_dxi_col = derivative_comm_part * w.t()[d:d+1,:]
            weighted_basis_value_dxi_col = basis_value_dxi_col * weights[j:end_index] 
            
            col_k_sym += weighted_basis_value_dxi_col.t() @ basis_value_dxi_col[:,neuron_num-1:neuron_num] # stifness matrix 
            col_k_nonsym += BETA* weighted_basis_value_col.t()@basis_value_dxi_col[:,neuron_num-1:neuron_num] # convection term (grad u, v)
            row_k_nonsym += BETA * weighted_basis_value_dxi_col.t() @ basis_value_col[:,neuron_num-1:neuron_num]

    # Neumman boundary condition
    if DIMENSION == 1: 
        if activation == 'relu':
            basis_value_col_bd = F.relu(integration_points_bd @ w.t()+ b)**(model.k) 
        elif activation == 'tanh':
            basis_value_col_bd = torch.tanh(integration_points_bd @ w.t()+ b) 
        weighted_basis_value_col_bd = basis_value_col_bd *weights_bd 
        dudn = g_N(integration_points_bd)* (torch.tensor([-1,1]).view(-1,1)).to(device) 
        rhs_gN =  (weighted_basis_value_col_bd[:,neuron_num-1:neuron_num]).t() @ dudn
    # neumann boundary condition 
    if DIMENSION > 1 and g_N != None:
        size_pts_bd = int(integration_points_bd.size(0)/(2*DIMENSION))
        bcs_N = g_N(DIMENSION)
        for ii, g_ii in bcs_N:
            #Another for loop needed if we need to divide the integration points into batches 
            weighted_g_N = -g_ii(integration_points_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:])* weights_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
            if activation == 'relu':
                basis_value_bd_col = F.relu(integration_points_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:] @ w.t()+ b)**(model.k)
            elif activation == 'tanh':
                basis_value_bd_col = torch.tanh(integration_points_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:] @ w.t()+ b) 
            rhs_gN += (basis_value_bd_col[:,neuron_num-1:neuron_num]).t() @ weighted_g_N

            weighted_g_N = g_ii(integration_points_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:])* weights_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:]
            if activation == 'relu':
                basis_value_bd_col = F.relu(integration_points_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:] @ w.t()+ b)**(model.k)
            elif activation == 'tanh':
                basis_value_bd_col = torch.tanh(integration_points_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:] @ w.t()+ b)
            rhs_gN += (basis_value_bd_col[:,neuron_num-1:neuron_num]) @ weighted_g_N
        rhs_k += rhs_gN 

    ## form the linear system by adding the last column and row 
#     print("size col_k",col_k)
    Mat[:neuron_num,neuron_num-1] = col_k_sym.view(-1)  + col_k_nonsym.view(-1)#col 
    Mat[neuron_num-1,:neuron_num-1] =  col_k_sym[:-1,:].view(-1) + row_k_nonsym[:-1,:].view(-1) # row 
#     Mat[neuron_num-1:neuron_num,:neuron_num-1:neuron_num] += row_k_nonsym[-1,-1]
    rhs_vec[neuron_num-1] = rhs_k.view(-1) 

    jac = Mat[:neuron_num,:neuron_num]
    rhs = rhs_vec[:neuron_num,:] 
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
    ## update the solution 
    return sol 

# %%
def select_greedy_neuron_ind(relu_dict_parameters,my_model,target,weights, integration_points,g_N,weights_bd, integration_points_bd,k,activation = 'relu',memory = 2**29): 
    dim = integration_points.size(1) 
    M = integration_points.size(0)
    N0 = relu_dict_parameters.size(0)   
    neuron_num = my_model.fc2.weight.size(1) if my_model != None else 0

    output = torch.zeros(N0,1).to(device) 
    s_time = time.time()
    total_size2 = M*(neuron_num+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # integration points 
    residual_values = torch.zeros(M,1).to(device) 

    if my_model!= None:
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2
            residual_values[jj:end_index] += - target(integration_points[jj:end_index]) 
            residual_values[jj:end_index] += LAMBDA * my_model(integration_points[jj:end_index,:]).detach()
    else:  
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2
            residual_values[jj:end_index] += - target(integration_points[jj:end_index])
    weight_func_values = residual_values*weights


    total_size = M * N0 
    num_batch = total_size//memory + 1 
    batch_size_1 = N0//num_batch # dictionary elements
    print("======argmax subproblem:f and N(u) terms, num batches: ",num_batch)
    for j in range(0,N0,batch_size_1):
        end_index = j + batch_size_1 
        if activation == 'relu':
            basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])**k) # uses broadcasting
        elif activation == 'tanh':
            basis_values = (torch.tanh( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim]))
        output[j:end_index] += basis_values.t()@weight_func_values #
    print('======TIME=======f and N(u) terms time :',time.time()-s_time)

    s_time =time.time() 
    if my_model!= None:
        #compute the derivative of the model 
        model_derivative_values = torch.zeros(M,dim).to(device) 
        for d in range(DIMENSION): ## there is a more efficient way 
            for jj in range(0,M,batch_size_2):
                end_index = jj + batch_size_2 
                model_derivative_values[jj:end_index,d:d+1] = my_model.evaluate_derivative(integration_points[jj:end_index,:],d+1).detach()
            #compute the derivative of the dictionary elements 
        for j in range(0,N0,batch_size_1): 
            end_index = j + batch_size_1 
            if activation == 'relu' and my_model.k == 1: 
                weighted_derivative_part = weights * torch.heaviside(integration_points@ (relu_dict_parameters[j:end_index,0:dim].T) - relu_dict_parameters[j:end_index,dim], ZERO)
                weighted_basis_value_col = weights *  (F.relu( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])**k) # uses broadcasting
            elif activation == 'relu' and my_model.k > 1:
                weighted_derivative_part = weights * my_model.k * F.relu(integration_points@ (relu_dict_parameters[j:end_index,0:dim].T) - relu_dict_parameters[j:end_index,dim])**(my_model.k-1)
                weighted_basis_value_col = weights *  (F.relu( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])**k) # uses broadcasting
            elif activation == 'tanh':
                weighted_derivative_part = weights * (1/torch.cosh(integration_points@ (relu_dict_parameters[j:end_index,0:dim].T) - relu_dict_parameters[j:end_index,dim])**2)
                weighted_basis_value_col = weights *  (torch.tanh( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])) # uses broadcasting
            for d in range(DIMENSION):
                weighted_basis_value_dx_col = weighted_derivative_part * relu_dict_parameters.t()[d:d+1,j:end_index] 
                output[j:end_index] += weighted_basis_value_dx_col.t() @ model_derivative_values[:,d:d+1]  # diffusion term
                output[j:end_index] += BETA * weighted_basis_value_col.t() @ model_derivative_values[:,d:d+1] # convection term 
    print("======argmax subproblem:< grad u_n, grad g> terms, num batches: ",num_batch)
    print('======TIME=======< grad u_n, grad g> terms time :',time.time()-s_time)
    
    
    # Neumann boundary condition
    s_time =time.time()  
    if g_N != None:  
        if DIMENSION == 1:
            if activation == 'relu':
                basis_values_bd_col = (F.relu(relu_dict_parameters[:,0] *integration_points_bd - relu_dict_parameters[:,1])**k) 
            elif activation == 'tanh':
                basis_values_bd_col = (torch.tanh(relu_dict_parameters[:,0] *integration_points_bd - relu_dict_parameters[:,1])) 
            weighted_basis_value_col_bd = basis_values_bd_col * weights_bd
            dudn = g_N(integration_points_bd)* (torch.tensor([-1,1]).view(-1,1)).to(device)
            output -=  weighted_basis_value_col_bd.t() @ dudn
        else: 
            size_pts_bd = int(integration_points_bd.size(0)/(2*DIMENSION)) # pre-defined rules for integration points on bdries
            bcs_N = g_N(dim)
            for ii, g_ii in bcs_N:
                # pts_bd_ii = pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
                weighted_g_N = -g_ii(integration_points_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:])* weights_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
                basis_value_bd_col = F.relu(integration_points_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim] )**(k)
                output -= basis_value_bd_col.t() @ weighted_g_N

                weighted_g_N = g_ii(integration_points_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:])* weights_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:]
                basis_value_bd_col = F.relu(integration_points_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:] @ (relu_dict_parameters[:,0:dim].T) - relu_dict_parameters[:,dim])**(k)
                output -= basis_value_bd_col.t() @ weighted_g_N
    print('======TIME=======Neumann boundary condition time :',time.time()-s_time)
    output = torch.abs(output) 
    neuron_index = torch.argmax(output.flatten())
    
    return neuron_index 

def OGAGeneralEllipticReLUNDim(my_model,target,u_exact,u_exact_grad,g_N, N_list,num_epochs,plot_freq = 10,Nx = 1024,order =5, activation = 'relu',k = 1,rand_deter = 'deter', solver = 'direct',memory = 2**29): 
    """ Orthogonal greedy algorithm to solve a general elliptic PDE over [0,1]^d
    two choices of activation: tanh, relu_k 
    """

    if DIMENSION == 1:
        weights, integration_points = PiecewiseGQ1D_weights_points(x_l= 0,x_r=1, Nx = Nx,order =order)
    elif DIMENSION == 2:
        weights, integration_points = PiecewiseGQ2D_weights_points(Nx = Nx, order = order)
    elif DIMENSION == 3:
        weights, integration_points = PiecewiseGQ3D_weights_points(Nx = Nx, order = order) 
    else:
        weights, integration_points = MonteCarlo_Sobol_dDim_weights_points(M = 2**14 ,d = 4)
    weights_bd, integration_points_bd = Neumann_boundary_quadrature_points_weights(M = 2**14,d = DIMENSION)
    M = integration_points.size(0) 

    # Compute initial L2 error and the gradient error 
    err = torch.zeros(num_epochs+1).to(device)
    err_h10 = torch.zeros(num_epochs+1).to(device)
    
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
        
    err[0] = compute_l2_error(u_exact,my_model,M,batch_size_2,weights,integration_points)
    err_h10[0] = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,weights,integration_points)

    start_time = time.time()
    solver = "direct"
    print("using linear solver: ",solver)
    
    N0 = np.prod(N_list) 
    dict_parameters = None 

    Mat = torch.zeros(num_epochs,num_epochs).to(device)  # size of the final matrix 
    rhs_vec = torch.zeros(num_epochs,1).to(device) # size of the final vector 

    for i in range(num_epochs): 
        print('epoch: ',i+1)

        if (rand_deter == 'deter' and i == 0) or (rand_deter == 'rand'): 
            dict_parameters = select_discrete_dictionary(activation,rand_deter,N_list,R = 0.4)
    
        neuron_index = select_greedy_neuron_ind(dict_parameters,my_model,target,weights, integration_points,g_N,weights_bd, integration_points_bd, k,activation = activation, memory=memory) 
        
        list_w.append(dict_parameters[neuron_index,0:DIMENSION])
        list_b.append(-dict_parameters[neuron_index,DIMENSION]) # different sign convention 
        num_neuron += 1
        if activation == 'relu':
            my_model = model(DIMENSION,num_neuron,1,k).to(device)
        elif activation == 'tanh':
            my_model = model_tanh(DIMENSION,num_neuron,1).to(device) 

        w_tensor = torch.stack(list_w, 0) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

#         sol = minimize_linear_layer_general_elliptic(my_model,target,weights, integration_points,weights_bd, integration_points_bd,g_N,activation =activation, solver = solver,memory = memory)
        sol = minimize_linear_layer_efficient(Mat, rhs_vec, my_model,target,weights, integration_points,weights_bd, integration_points_bd,g_N,activation =activation, solver = solver,memory = memory)
        sol = sol.flatten() 
        my_model.fc2.weight.data[0,:] = sol[:]

        #plot the solution 
        if DIMENSION == 1 and (i+1)%plot_freq == 0:  
            x_test = torch.linspace(0,1,200).view(-1,1).to(device)
            u_true = u_exact(x_test)
            plot_solution_modified(0,1,my_model,x_test,u_true)

        # Get L2 error and gradient error 
        total_size2 = M*(num_neuron+1)
        num_batch2 = total_size2//memory + 1 
        batch_size_2 = M//num_batch2 # integration points 
        err[i+1] = compute_l2_error(u_exact,my_model,M,batch_size_2,weights,integration_points)
        err_h10[i+1] = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,weights,integration_points)

    print("time taken: ",time.time() - start_time)
    return err.cpu(), err_h10.cpu(), my_model
