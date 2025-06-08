from l2min_oga2d_utils import * 

"""
L2 minimization, relu2, cos(pi*x1) * cos(pi*x2) in 2D, neuron number 2048.
Will be used for plotting the parameter distribution on a sphere.
"""

def target(x):
    return torch.cos(pi * x[:,0:1]) * torch.cos(pi * x[:,1:2]) 

dim =2 
function_name = "cosine" 
filename_write = "data_parameter_distribution_compare/2DQMCOGA-{}-order.txt".format(function_name)
Nx = 50
order = 3
relu_k = 2 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("relu {}, Integration points: Nx {}, order {} \n".format(relu_k, Nx, order))
f_write.close() 

save = True 
write2file = True 

for N0 in [2**11]: 
    save = True 
    f_write = open(filename_write, "a")
    my_model = None 
    s = 2**0
    Nx = 50 
    order = 3    
    exponent = 11 
    num_epochs = 2**exponent  
    plot_freq = num_epochs
    
    err_QMC2, my_model = OGAL2FittingReLU2D_QMC(my_model,target,s,N0,num_epochs,plot_freq, Nx, order, k =relu_k, linear_solver = "direct")
    
    if save: 
        folder = 'data_parameter_distribution_compare/'
        filename = folder + 'err_OGA_2D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(err_QMC2,filename) 
        folder = 'data_parameter_distribution_compare/'
        filename = folder + 'model_OGA_2D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(my_model.cpu().state_dict(),filename)

    show_convergence_order(err_QMC2,exponent,s*N0,relu_k,dim,filename_write,write2file = write2file)
    show_convergence_order_latex(err_QMC2,exponent,k=relu_k,d=dim)
