from neumann_oga3d_utils import * 
def run_trials():
    
    ## this is the example with oscillatory coefficient alpha 
    
    def u_exact(x):
        return torch.cos(pi*x[:,0:1])*torch.cos( pi*x[:,1:2]) * torch.cos(pi*x[:,2:3]) 
    def alpha(x): 
    #     return torch.ones(x.size(0),1).to(device)
        return 0.5 * torch.sin(6 * pi*x[:,0:1]) + 1. 
    
    def u_exact_grad():
        d = 3 
    
        def grad_1(x):
            return - pi* torch.sin(pi*x[:,0:1])*torch.cos( pi*x[:,1:2]) * torch.cos(pi*x[:,2:3])   
        def grad_2(x):
            return - pi* torch.cos(pi*x[:,0:1])*torch.sin( pi*x[:,1:2]) * torch.cos(pi*x[:,2:3])  
        def grad_3(x):
            return - pi* torch.cos(pi*x[:,0:1])*torch.cos( pi*x[:,1:2]) * torch.sin(pi*x[:,2:3])   
    
        u_grad=[grad_1, grad_2,grad_3] 
    
        return u_grad
    
    def target(x):
    
        z_c = torch.cos( pi*x[:,0:1])*torch.cos( pi*x[:,1:2] ) * torch.cos(pi*x[:,2:3]) 
        z1 = 3 * pi**2 * torch.sin(pi * x[:,0:1]) * torch.cos( 6*pi*x[:,0:1] ) * torch.cos( pi*x[:,1:2] )* torch.cos(pi*x[:,2:3]) 
        z2 = 0.5 * pi**2 * torch.sin(6*pi * x[:,0:1])* z_c 
        z = z1 + z2 + 2/2*pi**2 * torch.sin(6 * pi * x[:,0:1]) * z_c 
        z += ( 3 * (pi)**2 + 1)*z_c 
        return z 
    
    g_N = None 
    
    dim = 3 
    function_name = "cospix-osci-coef" 
    filename_write = "data-revision/3DOGA-{}-order.txt".format(function_name)
    Nx = 50 
    order = 3   
    relu_k = 3
    f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
    f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
    f_write.close() 
    save = True
    write2file = True
    rand_deter = 'rand'
    
    trial_num = 5
    for N_list in [[2**3,2**3,2**3]]: 
        for trial in range(trial_num):
            f_write = open(filename_write, "a")
            my_model = None 
            exponent = 10
            num_epochs = 2**exponent  
            plot_freq = num_epochs 
            N = np.prod(N_list)
            err_QMC2, err_h10, my_model = OGANeumannReLU3D(my_model,alpha, target,g_N, u_exact,u_exact_grad, N_list,num_epochs,plot_freq, Nx = Nx, order = order, k = relu_k, rand_deter= rand_deter, linear_solver = "direct")
            
            if save: 
                folder = 'data-revision/'
                filename = folder + 'errl2_NeumannOGA_OsciCoeff_3D_{}_relu{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
                torch.save(err_QMC2,filename) 
                filename = folder + 'errh10_NeumannOGA_OsciCoeff_3D_{}_relu{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
                torch.save(err_h10,filename) 
                filename = folder + 'model_NeumannOGA_OsciCoeff_3D_{}_relu{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name,relu_k,num_epochs,N,trial)
                torch.save(my_model.state_dict(),filename)
    
            show_convergence_order2(err_QMC2,err_h10,exponent,N,relu_k,dim,filename_write,write2file = write2file)
            show_convergence_order_latex2(err_QMC2,err_h10,exponent,k=relu_k,d = dim)


if __name__ == "__main__":
    run_trials()
