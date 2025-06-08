from l2min_oga10d_utils import * 

def target(x): ## Gaussian function in dimension 10  
    d = 10 
    cn =  7.03/d 
    return torch.exp(-torch.sum( cn**2 * (x - 0.5)**2,dim = 1, keepdim = True)) 

dim = 10 
function_name = "10DGaussian"
filename_write = "data-10d/1DOGA-{}-order.txt".format(function_name)
M = 2**20 # around2**19 50w 
relu_k = 4 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("relu {}, Integration points Quasi Monte Carlo:  {} \n".format(relu_k, M))
f_write.close() 
save = True 
write2file = True 
memory = 2**30 

s = 1 
for N0 in [2**7,2**8,2**9,2**10,2**11,2**12,2**13]: 

    N = s*N0 
    exponent = 9    
    num_epochs=  2**exponent 
    my_model = None 
    err, my_model = OGAL2FittingReLU4Dplus_QMC(my_model,target, \
                s,N0,num_epochs, M, k = relu_k, linear_solver = "direct", memory=memory)

    if save: 
        folder = 'data-10d/'
        filename = folder + function_name + "_err_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(relu_k,s * N0,num_epochs)
        torch.save(err,filename)
        filename = folder + function_name +  "_model_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(relu_k,s * N0,num_epochs)
        torch.save(my_model.state_dict(),filename) 

    show_convergence_order(err,exponent,N,relu_k,dim,filename_write,write2file = write2file)
    show_convergence_order_latex(err,exponent,k=relu_k,d=dim)

