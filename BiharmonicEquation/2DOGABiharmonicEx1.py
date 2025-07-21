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


def minimize_linear_layer_H2_explicit_assemble_efficient(model,target,weights, integration_points,activation = 'relu',solver="direct" ):
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

    if activation == 'relu':
        assert model.k != 1, "k must not be 1"  
        basis_value_col = F.relu(integration_points @ w.t()+ b)**(model.k) 
        if model.k == 2:  
            dxx_basis_value_col = 2 * torch.heaviside(integration_points @ w.t()+ b, ZERO) * (w.t()[0:1,:])**2 
            dxy_basis_value_col = 2 * torch.heaviside(integration_points @ w.t()+ b, ZERO) * (w.t()[0:1,:])* (w.t()[1:2,:]) 
            dyy_basis_value_col = 2 * torch.heaviside(integration_points @ w.t()+ b,ZERO) * (w.t()[1:2,:])**2 
        else: 
            dxx_basis_value_col = model.k * (model.k -1) * F.relu(integration_points @ w.t()+ b)**(model.k-2) * (w.t()[0:1,:])**2 
            dxy_basis_value_col = model.k * (model.k -1) * F.relu(integration_points @ w.t()+ b)**(model.k-2) * (w.t()[0:1,:])* (w.t()[1:2,:]) 
            dyy_basis_value_col = model.k * (model.k -1)* F.relu(integration_points @ w.t()+ b)**(model.k-2) * (w.t()[1:2,:])**2  
    # elif activation == 'tanh': 
    #     basis_value_col = torch.tanh(integration_points @ w.t()+ b) 
    #     basis_value_dx_col = tanh_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = tanh_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:]
    # elif activation == 'gaussian':
    #     basis_value_col = Gaussian_activation(integration_points @ w.t()+ b)
    #     basis_value_dx_col = Gaussian_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = Gaussian_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:]
    # elif activation == 'cosine':
    #     basis_value_col = cosine_activation(integration_points @ w.t()+ b) 
    #     basis_value_dx_col = cosine_activation_dx(integration_points @ w.t()+ b) * w.t()[0:1,:]
    #     basis_value_dy_col = cosine_activation_dx(integration_points @ w.t()+ b) * w.t()[1:2,:] 

    weighted_basis_value_col = basis_value_col * weights 
    jac = weighted_basis_value_col.t() @ basis_value_col  # mass matrix 
    rhs = weighted_basis_value_col.t() @ (target(integration_points)) 
    print("assembling the mass matrix time taken: ", time.time()-start_time) 

    start_time = time.time() 
    weighted_dxx_basis_value_col = dxx_basis_value_col * weights
    jac += weighted_dxx_basis_value_col.t() @ dxx_basis_value_col 
    jac += 2 * ( (dxy_basis_value_col * weights).t() @  dxy_basis_value_col )
    jac += (dyy_basis_value_col * weights).t() @  dyy_basis_value_col
    print("assembling the stiffness matrix time taken: ", time.time()-start_time)   
#     jac = jac1 + jac2    
    
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
## 
import sympy as sp
import torch
import numpy as np

# Define symbolic variables and the function u_exact
x_sym, y_sym = sp.symbols('x y')
u_expr = (4 * x_sym * (x_sym - 1))**4 * (4 * y_sym * (y_sym - 1))**4

# Compute symbolic derivatives (e.g., first derivative with respect to x)
u_x_expr = sp.diff(u_expr, x_sym)
u_y_expr = sp.diff(u_expr, y_sym)
# Compute higher-order derivatives if needed:
u_xx_expr = sp.diff(u_expr, x_sym, 2)
u_yy_expr = sp.diff(u_expr, y_sym, 2)
u_xy_expr = sp.diff(u_expr, x_sym, y_sym)
u_xxxx_expr = sp.diff(u_expr, x_sym, 4)
u_yyyy_expr = sp.diff(u_expr, y_sym, 4)
u_xxyy_expr = sp.diff(u_expr, x_sym, 2,y_sym,2)



# Convert the symbolic expressions to functions using lambdify (returns NumPy arrays)
u_exact_sym_func = sp.lambdify((x_sym, y_sym), u_expr, modules='numpy')
u_x_sym_func     = sp.lambdify((x_sym, y_sym), u_x_expr, modules='numpy')
u_y_sym_func     = sp.lambdify((x_sym, y_sym), u_y_expr, modules='numpy')
u_xx_sym_func    = sp.lambdify((x_sym, y_sym), u_xx_expr, modules='numpy')
u_yy_sym_func    = sp.lambdify((x_sym, y_sym), u_yy_expr, modules='numpy')
u_xy_sym_func    = sp.lambdify((x_sym, y_sym), u_xy_expr, modules='numpy')
u_xxxx_sym_func    = sp.lambdify((x_sym, y_sym), u_xxxx_expr, modules='numpy')
u_yyyy_sym_func    = sp.lambdify((x_sym, y_sym), u_yyyy_expr, modules='numpy')
u_xxyy_sym_func    = sp.lambdify((x_sym, y_sym), u_xxyy_expr, modules='numpy')


# Define wrapper functions that accept PyTorch tensors as input and return torch tensors.
def u_exact(x_tensor):
    # Assume x_tensor is a tensor of shape (N, 2) where each row is (x, y)
    x_np = x_tensor.detach().cpu().numpy()
    # Evaluate the symbolic function using the first and second columns
    result_np = u_exact_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    # Convert result to a torch tensor, preserving the device and dtype of the input
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_x(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_x_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_y(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_y_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_xx(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_xx_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_yy(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_yy_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_xy(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_xy_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_xxxx(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_xxxx_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_yyyy(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_yyyy_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)

def u_xxyy(x_tensor):
    x_np = x_tensor.detach().cpu().numpy()
    result_np = u_xxyy_sym_func(x_np[:, 0], x_np[:, 1]).reshape(-1,1)
    return torch.from_numpy(np.array(result_np)).to(x_tensor.device).type(x_tensor.dtype)


def rhs(x_tensor):
    return u_xxxx(x_tensor) + u_yyyy(x_tensor) + 2 * u_xxyy(x_tensor) + u_exact(x_tensor) 


# %%
derivatives = {
    "u_x": u_x,
    "u_y": u_y,
    "u_xx": u_xx,
    "u_yy": u_yy,
    "u_xy": u_xy,
}

def OGABiharmonicReLU2D(my_model,target,u_exact,derivatives, N_list,num_epochs,plot_freq, Nx, order, k =1, rand_deter = 'deter', linear_solver = "direct"): 
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
    gw_expand, integration_points = PiecewiseGQ2D_weights_points(Nx, order)
    gw_expand = gw_expand.to(device)
    integration_points = integration_points.to(device)

    err = torch.zeros(num_epochs+1)
    errh1 = torch.zeros(num_epochs+1)
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
    # print(func_values_sqrd.size())
    # print(gw_expand.size()) 
    err[0]= torch.sum(func_values_sqrd*gw_expand)**0.5
    errh1[0] = (gw_expand.t()@ ( derivatives['u_x'](integration_points)**2 + derivatives['u_y'](integration_points)**2 ))**0.5
    errh2[0] = (gw_expand.t()@ ( derivatives['u_xx'](integration_points)**2 \
                                + derivatives['u_yy'](integration_points)**2 \
                                + 2 * derivatives['u_xy'](integration_points)**2 ))**0.5
    start_time = time.time()
    solver = linear_solver

    N0 = np.prod(N_list)
    if rand_deter == 'deter':
        relu_dict_parameters = generate_relu_dict2D(N_list).to(device)
    print("using linear solver: ",solver)
    for i in range(num_epochs): 
        print("epoch: ",i+1, end = '\t')
        if rand_deter == 'rand':
            relu_dict_parameters = generate_relu_dict4plusD_sphere(2,1,N0).to(device)  
        if num_neuron == 0: 
            func_values = - target(integration_points)
        else: 
            func_values = - target(integration_points) + my_model(integration_points).detach()

        weight_func_values = func_values*gw_expand  
        basis_values = (F.relu( torch.matmul(integration_points,relu_dict_parameters[:,0:2].T ) - relu_dict_parameters[:,2])**k).T # uses broadcasting
        
        assert k != 1, "k must not be 1"  
        if k == 2:  
            dxx_basis_values = 2 * torch.heaviside(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2], ZERO) * (relu_dict_parameters.t()[0:1,:])**2  
            dyy_basis_values = 2 * torch.heaviside(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2], ZERO) * (relu_dict_parameters.t()[1:2,:])**2 
            dxy_basis_values = 2 * torch.heaviside(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2], ZERO) * (relu_dict_parameters.t()[0:1,:]) * (relu_dict_parameters.t()[1:2,:]) 
            if my_model!= None:
                dxx_my_model = my_model.evaluate_2ndderivative(integration_points,1,1).detach()
                dyy_my_model = my_model.evaluate_2ndderivative(integration_points,2,2).detach() 
                dxy_my_model = my_model.evaluate_2ndderivative(integration_points,1,2).detach() 
        else:  
            dxx_basis_values = k *(k-1) * F.relu(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2])**(k-2) * (relu_dict_parameters.t()[0:1,:])**2 
            dyy_basis_values = k *(k-1) * F.relu(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2])**(k-2) * (relu_dict_parameters.t()[1:2,:])**2 
            dxy_basis_values = k *(k-1) * F.relu(integration_points @ (relu_dict_parameters[:,0:2].T) - relu_dict_parameters[:,2])**(k-2) * (relu_dict_parameters.t()[0:1,:]) * (relu_dict_parameters.t()[1:2,:])
            if my_model!= None:
                dxx_my_model = my_model.evaluate_2ndderivative(integration_points,1,1).detach()
                dyy_my_model = my_model.evaluate_2ndderivative(integration_points,2,2).detach() 
                dxy_my_model = my_model.evaluate_2ndderivative(integration_points,1,2).detach() 


        output1 = torch.matmul(basis_values,weight_func_values) #
        if my_model!= None:
            output2 = dxx_basis_values.t() @(dxx_my_model*gw_expand) 
            output2 += 2 * dxy_basis_values.t() @ (dxy_my_model*gw_expand)
            output2 += dyy_basis_values .t() @(dyy_my_model*gw_expand)
            output = torch.abs(output1 + output2) 
        else: 
            output = torch.abs(output1) 
        # output = torch.abs(torch.matmul(basis_values,weight_func_values)) # 
        neuron_index = torch.argmax(output.flatten())
        
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:2]) # 
        list_b.append(-relu_dict_parameters[neuron_index,2])
        num_neuron += 1
        my_model = model(2,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        #Todo Done 
        sol = minimize_linear_layer_H2_explicit_assemble_efficient(my_model,target,gw_expand, integration_points,activation = 'relu',solver = solver)

        my_model.fc2.weight.data[0,:] = sol[:]
        # if (i+1)%plot_freq == 0: 
        #     plot_2D(my_model.cpu())
        #     my_model = my_model.to(device)

        model_values = my_model(integration_points).detach()
        # func_values = target(integration_points) - model_values
        # func_values_sqrd = func_values*func_values

        # L2 error ||u - u_n||
        diff_values_sqrd = (u_exact(integration_points) - model_values)**2 
        err[i+1]= torch.sum(diff_values_sqrd*gw_expand)**0.5
        errh1[i+1] += (gw_expand.t() @ (derivatives['u_x'](integration_points) - my_model.evaluate_derivative(integration_points,1).detach())**2).item()
        errh1[i+1] += (gw_expand.t() @ (derivatives['u_y'](integration_points) - my_model.evaluate_derivative(integration_points,2).detach())**2).item()
        errh1[i+1] = errh1[i+1]**0.5 
        errh2[i+1] += (gw_expand.t() @ (derivatives['u_xx'](integration_points) - my_model.evaluate_2ndderivative(integration_points,1,1).detach() )**2).item()
        errh2[i+1] += (2 * gw_expand.t() @ (derivatives['u_xy'](integration_points) - my_model.evaluate_2ndderivative(integration_points,1,2).detach() )**2).item()
        errh2[i+1] += (gw_expand.t() @ (derivatives['u_yy'](integration_points) - my_model.evaluate_2ndderivative(integration_points,2,2).detach() )**2).item()
        errh2[i+1] = errh2[i+1]**0.5
    print("time taken: ",time.time() - start_time)
    return err, errh1,errh2, my_model

def generate_relu_dict2D(N_list):
    N1 = N_list[0] 
    N2 = N_list[1]
    
    theta = np.linspace(0, 2*pi, N1, endpoint= False).reshape(N1,1)
    W1 = np.cos(theta)
    W2 = np.sin(theta)
    W = np.concatenate((W1,W2),1) # N1 x 2
    b = np.linspace(-1.42, 1.42, N2,endpoint=False).reshape(N2,1)
    
    index1 = np.arange(N1)
    index2 = np.arange(N2)
    ordered_pairs = np.array(np.meshgrid(index1,index2,indexing='ij'))

    ordered_pairs = ordered_pairs.reshape(2,-1).T
    W = W[ordered_pairs[:,0],:]
    b = b[ordered_pairs[:,1],:]
    Wb = np.concatenate((W,b),1) # N1 x 3 
    Wb_tensor = torch.from_numpy(Wb) 
    return Wb_tensor

# N_list = [10,20]
# Wb = generate_relu_dict2D(N_list).to(device)
# print(Wb.shape)

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
filename_write = "2DOGA-{}-order.txt".format(function_name)
f_write = open(filename_write, "a")
f_write.write("\n")
f_write.close() 
save = True 
relu_k= 3 

trial_num = 5 
for trial in range(trial_num): 
    for N_list in [[2**5,2**5]]: # ,[2**6,2**6],[2**7,2**7] 
        # save = True 
        f_write = open(filename_write, "a")
        my_model = None 
        Nx = 400
        order = 2   
        exponent = 9
        num_epochs = 2**exponent  
        plot_freq = num_epochs 
        N = np.prod(N_list)
        errl2,errh1,errh2, my_model = OGABiharmonicReLU2D(my_model,rhs,u_exact,derivatives, N_list,num_epochs,plot_freq, Nx, order, k = relu_k, rand_deter= 'rand', linear_solver = "direct")
        
        if save: 
            folder = 'data-biharmonic-2d-relu3/'
            filename = folder + 'err_OGA_2D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(errl2,filename) 
            filename = folder + 'errh2_OGA_2D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(errh2,filename)
            folder = 'data-biharmonic-2d-relu3/'
            filename = folder + 'model_OGA_2D_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(my_model.state_dict(),filename)

    show_convergence_order(errh1,errh2,exponent,N, filename_write,write2file = True)


# %%



