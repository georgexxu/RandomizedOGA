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


torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi)
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


def plot_subdomains(my_model):
    x_coord =torch.linspace(0,1,200)
    wi = my_model.fc1.weight.data
    bi = my_model.fc1.bias.data 
    for i, bias in enumerate(bi):  
        if wi[i,1] !=0: 
            plt.plot(x_coord, - wi[i,0]/wi[i,1]*x_coord - bias/wi[i,1])
        else: 
            plt.plot(x_coord,  - bias/wi[i,0]*torch.ones(x_coord.size()))

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.show()
    return 0   

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
    return Wb_tensor

def generate_relu_dict4D_sphere(s,N0): 
    samples = torch.randn(s*N0,5) 
    samples = samples/samples.norm(dim=1,keepdim=True)  
    Wb = samples 
    return Wb 

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

def minimize_linear_layer_H1_explicit_assemble_efficient(model,alpha, target, g_N, weights, integration_points, w_bd, pts_bd, activation = 'relu',solver="direct" ):
    """ -div alpha grad u(x) + u = f 
    Parameters
    ----------
    model: 
        nn model
    alpha:
        alpha function
    target:
        rhs function f 
    pts_bd:
        integration points on the boundary, embdedded in the domain 
    """ 

    start_time = time.time() 
    w = model.fc1.weight.data 
    b = model.fc1.bias.data 
    neuron_num = b.size(0) 
    dim = integration_points.size(1) 
    
    coef_alpha = alpha(integration_points) # alpha  
    if activation == 'relu':
        basis_value_col = F.relu(integration_points @ w.t()+ b)**(model.k)
        weighted_basis_value_col = basis_value_col * weights 
        jac = weighted_basis_value_col.t() @ basis_value_col  # mass matrix 
        rhs = weighted_basis_value_col.t() @ (target(integration_points)) 

        # Todo1: assemble the boundary condition term <g,v>_{\Gamma_N} 
        size_pts_bd = int(pts_bd.size(0)/(2*dim))
        if g_N != None:
            bcs_N = g_N(dim)
            for ii, g_ii in bcs_N:
                # pts_bd_ii = pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
                weighted_g_N = -g_ii(pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:])* w_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
                basis_value_bd_col = F.relu(pts_bd[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:] @ w.t()+ b)**(model.k)
                rhs += basis_value_bd_col.t() @ weighted_g_N

                weighted_g_N = g_ii(pts_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:])* w_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:]
                basis_value_bd_col = F.relu(pts_bd[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:] @ w.t()+ b)**(model.k)
                rhs += basis_value_bd_col.t() @ weighted_g_N
        
        if model.k == 1:  
            for d in range(dim):
                basis_value_dxi_col = torch.heaviside(integration_points @ w.t()+ b, ZERO) * w.t()[d:d+1,:]
                weighted_basis_value_dx_col = basis_value_dxi_col * weights * coef_alpha 
                jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 
            
        else: 
            for d in range(dim):
                basis_value_dxi_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[d:d+1,:]
                weighted_basis_value_dx_col = basis_value_dxi_col * weights * coef_alpha  
                jac += weighted_basis_value_dx_col.t() @ basis_value_dxi_col 
#             basis_value_dx_all_col = torch.stack([model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[d:d+1,:] for d in range(dim)]) 
            # basis_value_dx_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[0:1,:]
            # basis_value_dy_col = model.k * F.relu(integration_points @ w.t()+ b)**(model.k-1) * w.t()[1:2,:] 

    print("assembling the mass matrix time taken: ", time.time()-start_time) 


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


        ## argmax select greedy neuron part
def select_greedy_neuron_ind(relu_dict_parameters,my_model, alpha, target,gw_expand, integration_points,g_N,gw_expand_bd_faces, integration_points_bd_faces,k,memory=2**28): 

    #=================================================================
    dim = integration_points.size(1) 
    M = integration_points.size(0)
    N0 = relu_dict_parameters.size(0)   
    neuron_num = my_model.fc2.weight.size(1) if my_model != None else 0
    total_size2 = M*(neuron_num+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # integration points 

    if my_model != None: 
        func_values = - target(integration_points) 
        for jj in range(0,M,batch_size_2): 
            end_index = jj + batch_size_2 
            model_values = my_model(integration_points[jj:end_index,:]).detach()
            func_values[jj:end_index,:] += model_values #Change 1.  
    else: 
        func_values = - target(integration_points)    
    weight_func_values = func_values*gw_expand  

    total_size = M * N0 
    num_batch = total_size//memory + 1 
    batch_size = N0//num_batch

    output = torch.zeros(N0,1).to(device)
    print("argmax batch num, ", num_batch) 
    
    for j in range(0,N0,batch_size):  
        end_index = j + batch_size  
        basis_values_batch = (F.relu( torch.matmul(integration_points,relu_dict_parameters[j:end_index,0:dim].T ) - relu_dict_parameters[j:end_index,dim])**k).T # uses broadcasting    
        output[j:end_index,:]  = torch.matmul(basis_values_batch,weight_func_values)[:,:]

    # Gradient term: <alpha \nabla u_n, \nabla g_i>, i = 1,2,3,...,N
    alpha_coef = alpha(integration_points) # alpha 
    if my_model!= None:

        #compute the derivative of the model 
        model_derivative_values = torch.zeros(M,dim).to(device) 
        for d in range(dim): ## there is a more efficient way 
            for jj in range(0,M,batch_size_2):
                end_index = jj + batch_size_2 
                model_derivative_values[jj:end_index,d:d+1] = my_model.evaluate_derivative(integration_points[jj:end_index,:],d+1).detach()
                
        if k == 1:  
            for j in range(0,N0,batch_size):  
                end_index = j + batch_size 
                derivative_part = torch.heaviside(integration_points @ (relu_dict_parameters[j:end_index,0:dim].T) - relu_dict_parameters[j:end_index,dim], ZERO) # dimension 4 
                derivative_part *= alpha_coef # alpha 
                for dx_i in range(dim): 

                    weight_dbasis_values_dxi =  (derivative_part * relu_dict_parameters.t()[dx_i:dx_i+1,j:end_index]) *gw_expand   
                    # dmy_model_dxi = my_model.evaluate_derivative(integration_points,dx_i+1).detach()
                    output[j:end_index,:] += torch.matmul(weight_dbasis_values_dxi.t(), model_derivative_values[:,dx_i:dx_i+1]) 

        else:  
            for j in range(0,N0,batch_size):  
                end_index = j + batch_size 
                derivative_part = k * F.relu(integration_points @ (relu_dict_parameters[j:end_index,0:dim].T) - relu_dict_parameters[j:end_index,dim])**(k-1) # dimension 4 
                derivative_part *= alpha_coef # alpha
                for dx_i in range(dim): 

                    weight_dbasis_values_dxi =  (derivative_part * relu_dict_parameters.t()[dx_i:dx_i+1,j:end_index]) * gw_expand    
                    # dmy_model_dxi = my_model.evaluate_derivative(integration_points,dx_i+1).detach()
                    output[j:end_index,:] += torch.matmul(weight_dbasis_values_dxi.t(), model_derivative_values[:,dx_i:dx_i+1]) 


    #Boundary condition term -<g,v>_{\Gamma_N}  
    if g_N != None:
        bcs_N = g_N(dim) 
        size_pts_bd = int(integration_points_bd_faces.size(0)/(2*dim)) # pre-defined rules for integration points on bdries
        for ii, g_ii in bcs_N: 
            
            weighted_g_N = -g_ii(integration_points_bd_faces[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:])* gw_expand_bd_faces[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:]
            basis_values_bd_faces = (F.relu( torch.matmul(integration_points_bd_faces[2*ii*size_pts_bd:(2*ii+1)*size_pts_bd,:],relu_dict_parameters[:,0:dim].T ) - relu_dict_parameters[:,dim])**k).T
            output -= torch.matmul(basis_values_bd_faces,weighted_g_N)
            
            weighted_g_N = g_ii(integration_points_bd_faces[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:])* gw_expand_bd_faces[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:]
            basis_values_bd_faces = (F.relu( torch.matmul(integration_points_bd_faces[(2*ii+1)*size_pts_bd:(2*ii+2)*size_pts_bd,:],relu_dict_parameters[:,0:dim].T ) - relu_dict_parameters[:,dim])**k).T
            output -= torch.matmul(basis_values_bd_faces,weighted_g_N)

    output = torch.abs(output)
    neuron_index = torch.argmax(output.flatten())
    return neuron_index 
    

def OGANeumannReLU4D(my_model,alpha, target,g_N, u_exact, u_exact_grad, N_list,num_epochs,plot_freq, M, k =1, rand_deter = 'deter', linear_solver = "direct",memory = 2**29): 
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
    dim = 4 
    gw_expand, integration_points = MonteCarlo_Sobol_dDim_weights_points(M, d=dim)
    gw_expand = gw_expand.to(device)
    integration_points = integration_points.to(device)

    # define integration on the boundary 
    gw_expand_bd, integration_points_bd = PiecewiseGQ3D_weights_points(25, order = 2) 
    gw_expand_bd_faces, integration_points_bd_faces = Neumann_boundary_quadrature_points_weights(99999999,dim) 

    err = torch.zeros(num_epochs+1)
    err_h10 = torch.zeros(num_epochs+1).to(device) 
    num_neuron = 0 if my_model == None else int(my_model.fc1.bias.detach().data.size(0))
    total_size2 = M*(num_neuron+1)
    num_batch2 = total_size2//memory + 1 
    batch_size_2 = M//num_batch2 # integration points 

    if my_model == None: 
        list_b,list_w = [],[]
    else:
        bias = my_model.fc1.bias.detach().data
        nnweights = my_model.fc1.weight.detach().data
        list_b,list_w = list(bias), list(nnweights)
        
    err[0] = compute_l2_error(u_exact,my_model,M,batch_size_2,gw_expand,integration_points)
    err_h10[0] = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,gw_expand,integration_points)
    
    start_time = time.time()
    solver = linear_solver
    N0 = np.prod(N_list)
    if rand_deter == 'deter':
        relu_dict_parameters = generate_relu_dict4D(N_list).to(device)
    print("using linear solver: ",solver)

    for i in range(num_epochs): 
        print("epoch: ",i+1, end = '\t')
        if rand_deter == 'rand':
            # relu_dict_parameters = generate_relu_dict4D_QMC(1,N0).to(device) 
            relu_dict_parameters = generate_relu_dict4D_sphere(1,N0).to(device)
        neuron_index = select_greedy_neuron_ind(relu_dict_parameters,my_model,alpha, target,gw_expand, integration_points,g_N,gw_expand_bd_faces, integration_points_bd_faces,k,memory=memory)

    
        # print(neuron_index)
        list_w.append(relu_dict_parameters[neuron_index,0:dim]) # dimension 4 
        list_b.append(-relu_dict_parameters[neuron_index,dim])
        num_neuron += 1
        my_model = model(dim,num_neuron,1,k).to(device)
        w_tensor = torch.stack(list_w, 0 ) 
        b_tensor = torch.tensor(list_b)
        my_model.fc1.weight.data[:,:] = w_tensor[:,:]
        my_model.fc1.bias.data[:] = b_tensor[:]

        sol = minimize_linear_layer_H1_explicit_assemble_efficient(my_model,alpha, target, g_N, gw_expand, integration_points,gw_expand_bd_faces, integration_points_bd_faces,activation = 'relu',solver = solver)

        my_model.fc2.weight.data[0,:] = sol[:]

        # compute L2 and H10 errors
        err[i+1] = compute_l2_error(u_exact,my_model,M,batch_size_2,gw_expand,integration_points)
        err_h10[i+1] = compute_gradient_error(u_exact_grad,my_model,M,batch_size_2,gw_expand,integration_points)
    
    print("time taken: ",time.time() - start_time)
    return err, err_h10.cpu(), my_model

