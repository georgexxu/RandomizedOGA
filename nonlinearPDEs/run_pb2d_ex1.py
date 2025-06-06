from nonlinear_oga_2d_utils import * 

sigma = 0.15
m = 4 

def u_exact(x):
    """Gabor function applied to a batch of points."""
    exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
    cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])  # x[:, 0:1] keeps the dimension
    return exp_term * cos_term


def u_exact_grad():
    def u_grad_x(x):
        exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
        cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])
        sin_term = torch.sin(2 * math.pi * m * x[:, 0:1])

        # Derivative with respect to x_1
        du_dx1 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * cos_term - (2 * math.pi * m * exp_term * sin_term)
        return du_dx1
        
    def u_grad_y(x):

        exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
        cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])
        sin_term = torch.sin(2 * math.pi * m * x[:, 0:1])
        du_dx_other = exp_term * (-2 * (x[:, 1:2] - 0.5) / (2 * sigma ** 2)) * cos_term
        return du_dx_other

    u_grad=[] 
    u_grad.append(u_grad_x)
    u_grad.append(u_grad_y)
    return u_grad


def laplace_u_exact(x):
    """Laplacian of the Gabor function for a batch of points, summed over all dimensions."""

    exp_term = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma ** 2))
    cos_term = torch.cos(2 * math.pi * m * x[:, 0:1])
    sin_term = torch.sin(2 * math.pi * m * x[:, 0:1])

    # Laplacian in x_1: Apply the product rule twice
    laplace_x1_part_1 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2))**2 * cos_term
    laplace_x1_part_2 = exp_term * ( - 1 / sigma ** 2) * cos_term
    laplace_x1_part_3 = exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * (-2*m*math.pi)*sin_term
    laplace_x1_part_4 = - (2 * math.pi * m * exp_term * (-2 * (x[:, 0:1] - 0.5) / (2 * sigma ** 2)) * sin_term) 
    laplace_x1_part_5 = - ( (2 * math.pi * m)**2 * exp_term * cos_term)

    laplace_x1 = laplace_x1_part_1 + laplace_x1_part_2 + laplace_x1_part_3 \
                + laplace_x1_part_4 + laplace_x1_part_5

    # Laplacian in other dimensions (x_i, i > 1): Only the exponential term matters here
    laplace_other = (-1 / sigma ** 2) * exp_term * cos_term \
                + exp_term * (-2 * (x[:, 1:] - 0.5) / (2 * sigma ** 2))**2 * cos_term 

    # Sum all Laplacians over each variable to get the total Laplacian
    laplace_sum = laplace_x1  + torch.sum(laplace_other, dim=1, keepdim=True)

    return laplace_sum 

def rhs(x):
    """Right-hand side of the equation using the Gabor function for a batch of points."""
    laplace_sum = laplace_u_exact(x)
    return -laplace_sum + nonlinear(u_exact(x))

def g_N(dim):
    u_grad = u_exact_grad() 
    bcs_N = []
    for i in range(dim):
        bcs_N.append((i, u_grad[i]))
    return bcs_N 

dim = 2 
function_name = "gabor2d-m4" 
filename_write = "data-revision1/2DOGA-{}-order.txt".format(function_name)
Nx = 600  
order = 3 
relu_k = 3 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
f_write.close() 
save = True 
write2file = True 
memory = 2**29 
rand_deter = 'rand'

trial_num = 5
for N_list in [[2**4,2**5]]: # ,[2**6,2**6],[2**7,2**7] 
    for trial in range(trial_num):
        f_write = open(filename_write, "a")
        my_model = None 
        exponent = 9 
        num_epochs = 2**exponent  
        plot_freq = num_epochs 
        N = np.prod(N_list)
        err_QMC2, err_h10, my_model = CGANonlinearPoissonReLU2D(my_model,rhs,u_exact, u_exact_grad,g_N, N_list,num_epochs,plot_freq, Nx, order, k = relu_k, rand_deter = rand_deter, linear_solver = "direct",plot=False,memory=memory)
        if save: 
            folder = 'data-revision1/'
            filename = folder + 'err_OGA_2D_{}_neuron_{}_N_{}_{}_trial_{}.pt'.format(function_name,num_epochs,N,rand_deter,trial)
            torch.save(err_QMC2,filename) 
            filename = folder + 'err_h10_OGA_2D_{}_neuron_{}_N_{}_{}_trial_{}.pt'.format(function_name,num_epochs,N,rand_deter,trial)
            torch.save(err_h10,filename) 
            filename = folder + 'model_OGA_2D_{}_neuron_{}_N_{}_{}_trial_{}.pt'.format(function_name,num_epochs,N,rand_deter,trial)
            torch.save(my_model.cpu().state_dict(),filename)

        show_convergence_order2(err_QMC2,err_h10,exponent,N,relu_k,dim,filename_write,write2file = write2file)
        show_convergence_order_latex2(err_QMC2,err_h10,exponent,relu_k,dim)
