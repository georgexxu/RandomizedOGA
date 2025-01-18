import sys 
sys.path.insert(0, '../')
from Greedy4DQMC_m import * 

def target(x):
    ## Gaussian function in 4d 
    d = 4 
    cn =   7.03/d 
    return torch.exp(-torch.sum( cn**2 * (x - 0.5)**2,dim = 1, keepdim = True)) 

save = True  
experiment_label = "ex2"
for k in [4]: 

    for N_list in [[2**4, 2**4, 2**3, 2**3],[2**3, 2**3, 2**3, 2**3],[2**3, 2**3, 2**2, 2**2]]: 
        N = np.prod(N_list) 
        print()
        print() 
        exponent = 9 
        num_epochs=  2**exponent 
        M = 2**19 # around 50w 
        print(M)
        my_model = None 
        err, my_model = OGAL2FittingReLU4D(my_model,target,N_list,num_epochs, M, k = k, linear_solver = "direct")

        if save: 
            filename = experiment_label + "_err_deterministic_Dict_relu_{}_size_{}_num_neurons_{}.pt".format(k,N,num_epochs)
            torch.save(err,filename)
            filename = experiment_label + "_model_deterministic_Dict_relu_{}_size_{}_num_neurons_{}.pt".format(k,N,num_epochs)
            torch.save(my_model.state_dict(),filename) 

        print_convergence_order(err,exponent) 

