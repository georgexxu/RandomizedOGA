from nonlinear_oga_3d_utils import * 

freq = 4 
sigma = 0.15 
def gaussian(x):
    return torch.exp(-torch.sum( (x - 0.5)**2,dim=1,keepdim=True)/(2 *sigma**2) ) 
def gaussian_grad_1(x):
    return  gaussian(x) * (- (x[:,0:1] - 0.5)/(sigma**2) ) 
def gaussian_grad_2(x):
    return  gaussian(x) * (- (x[:,1:2] - 0.5)/(sigma**2) ) 
def gaussian_grad_3(x):
    return  gaussian(x) * (- (x[:,2:3] - 0.5)/(sigma**2) ) 

def u_exact(x):
    return gaussian(x) * torch.cos(2*pi*freq*x[:,0:1]) 
def alpha(x): 
    return torch.ones(x.size(0),1).to(device)

def u_grad_1(x):
    return  torch.cos(2*pi*freq*x[:,0:1]) *gaussian_grad_1(x) \
            - 2*pi*freq * torch.sin(2*pi*freq*x[:,0:1]) * gaussian(x) 
def u_grad_2(x):
    return torch.cos(2*pi*freq*x[:,0:1]) * gaussian_grad_2(x)
def u_grad_3(x):
    return  torch.cos(2*pi*freq*x[:,0:1]) * gaussian_grad_3(x)

def u_exact_grad():
    d = 3 
    def u_grad_1(x):
        return  torch.cos(2*pi*freq*x[:,0:1]) *gaussian_grad_1(x) \
                - 2*pi*freq * torch.sin(2*pi*freq*x[:,0:1]) * gaussian(x) 
    def u_grad_2(x):
        return torch.cos(2*pi*freq*x[:,0:1]) * gaussian_grad_2(x)
    def u_grad_3(x):
        return  torch.cos(2*pi*freq*x[:,0:1]) * gaussian_grad_3(x)

    u_grad=[u_grad_1, u_grad_2,u_grad_3] 
    return u_grad

def laplace_u_exact(x):
    return - 2*pi*freq * torch.sin(2*pi*freq*x[:,0:1]) *gaussian_grad_1(x) \
            + torch.cos(2*pi*freq*x[:,0:1])*( gaussian(x) * ( ((x[:,0:1] - 0.5)/(sigma**2))**2 -1/(sigma**2))  ) \
            -( (2*pi*freq)**2 * torch.cos(2*pi*freq*x[:,0:1]) * gaussian(x) + (2*pi*freq)*torch.sin(2*pi*freq*x[:,0:1]) * gaussian_grad_1(x) ) \
            + torch.cos(2*pi*freq*x[:,0:1]) * (gaussian(x) * ( ((x[:,1:2] - 0.5)/(sigma**2))**2 -1/(sigma**2) )  ) \
            + torch.cos(2*pi*freq*x[:,0:1]) * ( gaussian(x) * ( ((x[:,2:3] - 0.5)/(sigma**2))**2 -1/(sigma**2) )   ) \

def target(x):
    return - laplace_u_exact(x) + nonlinear(u_exact(x)) 

def g_N(dim):
    u_grad = u_exact_grad() 
    bcs_N = []
    for i in range(dim):
        bcs_N.append((i, u_grad[i]))
    return bcs_N

def rhs(x):
    return  -laplace_u_exact(x) + nonlinear(u_exact(x)) 


dim = 3 
function_name = "gabor3d-m4" 
filename_write = "data/3DCGA-{}-order.txt".format(function_name)
Nx = 50
order = 2 
relu_k = 3
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
f_write.close() 
save = True 
write2file = True 
memory = 2**29 

trial_num = 5 
for N_list in [[2**3,2**3,2**3]]: # ,[2**6,2**6],[2**7,2**7] 
    for trial in range(trial_num): 
        f_write = open(filename_write, "a")
        my_model = None 
        exponent = 10  
        num_epochs = 2**exponent
        plot_freq = num_epochs 
        N = np.prod(N_list)
        err_QMC2, err_h10, my_model = CGANonlinearPoissonReLU3D(my_model,rhs,alpha, u_exact, u_exact_grad,g_N,\
                                            N_list,num_epochs,plot_freq, Nx, order, k = relu_k, \
                                            rand_deter = 'rand', linear_solver = "direct",memory = memory)

        if save: 
            folder = 'data-revision1/'

            filename = folder + 'errl2_OGA_3D_{}_relu_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
            torch.save(err_QMC2,filename) 
            filename = folder + 'errh10_OGA_3D_{}_relu_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
            torch.save(err_h10,filename) 
            filename = folder + 'model_OGA_3D_{}_relu_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
            torch.save(my_model.cpu().state_dict(),filename)

        show_convergence_order2(err_QMC2,err_h10,exponent,N,relu_k,dim,filename_write,write2file = write2file)
        show_convergence_order_latex2(err_QMC2,err_h10,exponent,k =relu_k,d = dim)

