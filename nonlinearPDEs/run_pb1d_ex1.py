from nonlinear_oga_1d_utils import * 


# Parameters for the Gabor function
sigma = 0.15
m = 4 

def u_exact(x):
    """Gabor function applied to a batch of points."""
    exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
    cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])  # x[:, 0:1] keeps the dimension
    return exp_term * cos_term

def du_exact(x):
    """First derivative of the Gabor function with respect to each component for a batch of points."""
    exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
    cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])
    sin_term = torch.sin(2 * math.pi * m * x[:, 0:1])

    # Derivative with respect to x_1
    du_dx1 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * cos_term - (2 * math.pi * m * exp_term * sin_term)

    # Derivative with respect to other dimensions
    du_dx_other = exp_term * (-2 * (x[:, 1:] - 0.5) / (2 * sigma ** 2)) * cos_term

    # Concatenate all the derivatives to keep dimension
    du_dx = torch.cat([du_dx1, du_dx_other], dim=1)

    return du_dx

def laplace_u_exact(x):
    """Laplacian of the Gabor function for a batch of points, summed over all dimensions."""
    exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
    cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])
    sin_term = torch.sin(2 * math.pi * m * x[:, 0:1])

    exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * cos_term 
    - (2 * math.pi * m * exp_term * sin_term) 
    # Laplacian in x_1: Apply the product rule twice
    laplace_x1_part_1 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2))**2 * cos_term
    laplace_x1_part_2 = exp_term * ( - 1 / sigma ** 2) * cos_term
    laplace_x1_part_3 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * (-2*m*math.pi)*sin_term
    laplace_x1_part_4 = - (2 * math.pi * m * exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * sin_term) 
    laplace_x1_part_5 = - ( (2 * math.pi * m)**2 * exp_term * cos_term)

    laplace_x1 = laplace_x1_part_1 + laplace_x1_part_2 + laplace_x1_part_3 \
                + laplace_x1_part_4 + laplace_x1_part_5

    # Laplacian in other dimensions (x_i, i > 1): Only the exponential term matters here
    laplace_other = (-1 / sigma ** 2) * exp_term * (((x[:, 1:] - 0.5) ** 2 / sigma ** 2) - 1) * cos_term \
                + exp_term * (-2 * (x[:, 1:] - 0.5) / (2 * sigma ** 2))**2 * cos_term 

    # Sum all Laplacians over each variable to get the total Laplacian
    laplace_sum = laplace_x1  + torch.sum(laplace_other, dim=1, keepdim=True)

    return laplace_sum 

def rhs(x):
    """Right-hand side of the equation using the Gabor function for a batch of points."""
    laplace_sum = laplace_u_exact(x)
    return -laplace_sum + nonlinear(u_exact(x))

def g_N(x):
    return du_exact(x) 

dim = 1
function_name = "gabor1d-m4" 
filename_write = "data/1DOGA-PBE-{}-order.txt".format(function_name)
Nx = 2**10
order = 3
relu_k = 3 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
f_write.close() 
save = True 
write2file = True 
memory = 2**29 

trial_num = 1
for N in [2**9]: # 2**12,2**14
    for trial in range(trial_num): 
        f_write = open(filename_write, "a")
        my_model = None 
        exponent = 5

        num_epochs = 2**exponent  
        plot_freq = num_epochs 

        err_QMC2, err_h10, my_model = OGARandPBEReLU1D(None,rhs,u_exact,du_exact,g_N, N,num_epochs,plot_freq, Nx, order, k =relu_k, solver = "direct")
        
        if save: 
            folder = 'data-revision1/'
            filename = folder + 'errl2_CGA_1D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(err_QMC2,filename) 
            filename = folder + 'errh10_CGA_1D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(err_h10,filename) 
            filename = folder + 'model_CGA_1D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(my_model.cpu().state_dict(),filename)
        
        show_convergence_order2(err_QMC2,err_h10,exponent,N,relu_k, dim, filename_write,write2file=write2file)
        show_convergence_order_latex2(err_QMC2,err_h10,exponent,k =relu_k, d = dim)


