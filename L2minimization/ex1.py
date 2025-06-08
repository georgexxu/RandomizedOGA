from l2min_oga2d_utils import * 


def target(x):
    sigma = 0.15# s.d of Gaussian 
    frequency = 8  # frequency of cosine 
    z=  torch.exp( - ( (x[:,0:1]-0.5)**2 + ( x[:,1:2] -0.5)**2)  / (2*sigma**2)) *  torch.cos(2 * pi * frequency * x[:,0:1])
    return z 

dim =2 
function_name = "gabor2d" 
filename_write = "data-2d/2DQMCOGA-{}-order.txt".format(function_name)
Nx = 50
order = 3
relu_k = 1  
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("relu {}, Integration points: Nx {}, order {} \n".format(relu_k, Nx, order))
f_write.close() 

save = True 
write2file = True 

for N0 in [2**6,2**7,2**8,2**9,2**10]: 
    save = True 
    f_write = open(filename_write, "a")
    my_model = None 
    s = 2**0
    Nx = 50 
    order = 3    
    exponent = 9  
    num_epochs = 2**exponent  
    plot_freq = num_epochs
    
    err_QMC2, my_model = OGAL2FittingReLU2D_QMC(my_model,target,s,N0,num_epochs,plot_freq, Nx, order, k =relu_k, linear_solver = "direct")
    
    if save: 
        folder = 'data-2d/'
        filename = folder + 'err_OGA_2D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(err_QMC2,filename) 
        folder = 'data-2d/'
        filename = folder + 'model_OGA_2D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(my_model,filename)

    show_convergence_order(err_QMC2,exponent,s*N0,relu_k,dim,filename_write,write2file = write2file)
    show_convergence_order_latex(err_QMC2,exponent,k=relu_k,d=dim)
