from l2min_oga3d_utils import * 

# %%
def target(x):
    return torch.sin(pi*x[:,0:1])*torch.sin(pi*x[:,1:2])*torch.sin(pi*x[:,2:3]) 

function_name = "sinpipipi" 
filename_write = "data-3d/3DQMCOGA-{}-order.txt".format(function_name)
dim = 3 
Nx = 25   
order = 3   
relu_k = 1  
save = True 
write2file = True 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("relu {}, Integration points: Nx {}, order {} \n".format(relu_k, Nx, order))
f_write.close() 

s = 2**0 
for N0 in [2**6,2**7,2**8,2**9,2**10]: 
    f_write = open(filename_write, "a")
    my_model = None 
    exponent = 9
    num_epochs = 2**exponent  
    plot_freq = 1 
    err_QMC2, my_model = OGAL2FittingReLU3D_QMC(my_model,target,s,N0,num_epochs,plot_freq, Nx, order, k =relu_k, linear_solver = "direct")
    
    if save: 
        folder = 'data-3d/'
        filename = folder + 'err_OGA_3D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(err_QMC2,filename) 
        folder = 'data-3d/'
        filename = folder + 'model_OGA_3D_{}_neuron_{}_N_{}_randomized.pt'.format(function_name,num_epochs,s*N0)
        torch.save(my_model,filename)
        
    show_convergence_order(err_QMC2,exponent,s*N0,relu_k,dim,filename_write,write2file = write2file)
    show_convergence_order_latex(err_QMC2,exponent,k=relu_k,d=dim)
    
# %%



