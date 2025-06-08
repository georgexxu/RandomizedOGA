from l2min_oga4d_utils import * 

"""
ex4, 4d product of sines, ReLU4 activation 
"""
def target(x):
    return torch.sin(pi*x[:,0:1])*torch.sin(pi*x[:,1:2])*torch.sin(pi*x[:,2:3]) *torch.sin(pi*x[:,3:4])

dim = 4 
function_name = "sin-product-4d" 
filename_write = "data-4d/4DQMCOGA-{}-order.txt".format(function_name)
M = 2**19 # MC points around 50w 
relu_k = 4 
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("relu {}, Integration points: {} \n".format(relu_k, M))
f_write.close() 

save = True 
write2file = True  
memory = 2**29 
s = 1 
for N0 in [2**6,2**7,2**8,2**9,2**10]: 
    N = s*N0
    exponent = 9  
    num_epochs=  2**exponent 
    my_model = None 
    err, my_model = OGAL2FittingReLU4D_QMC(my_model,target, \
                s,N0,num_epochs, M, k = relu_k, linear_solver = "direct", memory = memory)

    if save: 
        folder = 'data-4d/'
        filename = folder + function_name + "_err_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(relu_k,s * N0,num_epochs)
        torch.save(err,filename)
        filename = folder + function_name + "_model_randDict_relu_{}_size_{}_num_neurons_{}.pt".format(relu_k,s * N0,num_epochs)
        torch.save(my_model.state_dict(),filename) 
        
    show_convergence_order(err,exponent,N,relu_k, dim,filename_write, write2file = write2file)
    show_convergence_order_latex(err,exponent,k=relu_k,d=dim)

