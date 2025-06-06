from nonlinear_oga_1d_utils import * 

def u_exact(x):
    return torch.cos(2*pi*x)

def du_exact(x):
    return -2 *pi*torch.sin(2*pi*x)

def rhs(x):
    return  (2*pi)**2 * torch.cos(2*pi*x) + nonlinear(torch.cos(2*pi*x))

def g_N(x):
    return du_exact(x) 

dim = 1 
function_name = "sine1d" 
filename_write = "data-revision1/1DRandCGA-PBE-{}-order.txt".format(function_name)
Nx = 2**10 # 2**13
order = 3  
relu_k = 3   
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
f_write.close() 
save = True 
write2file = True 
memory = 2**29 

trial_num = 2
for N in [2**10]: 
    for trial in range(trial_num): 
        f_write = open(filename_write, "a")
        my_model = None 
        exponent = 5
        num_epochs = 2**exponent  
        plot_freq = num_epochs 

        err_QMC2, err_h10, my_model = OGARandPBEReLU1D(None,rhs,u_exact,du_exact,g_N, N,num_epochs,plot_freq, Nx, order, k =relu_k , solver = "direct")
        
        if save: 
            folder = 'data-revision1/'
            filename = folder + 'errl2_CGA_1D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,num_epochs,N,trial )
            torch.save(err_QMC2,filename) 
            filename = folder + 'model_CGA_1D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,num_epochs,N,trial)
            torch.save(my_model.cpu().state_dict(),filename)
        
        show_convergence_order2(err_QMC2,err_h10,exponent,N,filename_write,write2file)
        show_convergence_order_latex2(err_QMC2,err_h10,exponent, k = relu_k, d = dim )

