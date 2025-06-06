from nonlinear_oga_3d_utils import * 





# %%
freq = 1
def u_exact(x):
    return torch.cos(freq*pi*x[:,0:1])*torch.cos( freq*pi*x[:,1:2]) * torch.cos(freq*pi*x[:,2:3])  
def alpha(x): 
    return torch.ones(x.size(0),1).to(device)

def u_exact_grad():
    d = 3 
    def grad_1(x):
        return - freq*pi* torch.sin(freq*pi*x[:,0:1])*torch.cos( freq*pi*x[:,1:2]) * torch.cos(freq*pi*x[:,2:3])   
    def grad_2(x):
        return - freq*pi* torch.cos(freq*pi*x[:,0:1])*torch.sin( freq*pi*x[:,1:2]) * torch.cos(freq*pi*x[:,2:3])  
    def grad_3(x):
        return - freq*pi* torch.cos(freq*pi*x[:,0:1])*torch.cos( freq*pi*x[:,1:2]) * torch.sin(freq*pi*x[:,2:3])   
    
    u_grad=[grad_1, grad_2,grad_3] 

    return u_grad
def laplace_u_exact(x):
    return -3*(freq*pi)**2 * torch.cos(freq*pi*x[:,0:1])*torch.cos( freq*pi*x[:,1:2]) * torch.cos(freq*pi*x[:,2:3])

def u_exact_approx(x):
    return 0.7 * u_exact(x)

def rhs(x):
    return  -laplace_u_exact(x) + nonlinear(u_exact(x))
g_N = None 

dim = 3 
function_name = "cospix" 
filename_write = "data/3DCGA-{}-order.txt".format(function_name)
Nx = 50
order = 3
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

