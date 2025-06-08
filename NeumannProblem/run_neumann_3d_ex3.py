from neumann_oga3d_utils import * 

def run_trials():
    def u_exact(x):
        d = 3 
        cn =   7.03/d 
        return torch.exp(-torch.sum( cn**2 * (x - 0.5)**2,dim = 1, keepdim = True))  

    def u_exact_grad():
        d = 3 
        def make_grad_i(i):
            def grad_i(x):
                d = 3  
                cn = 7.03/d
                return torch.exp(-torch.sum(cn**2 * (x - 0.5)**2, dim=1, keepdim=True)) * (-2 * cn**2 * (x[:, i:i+1] - 0.5))
            return grad_i 
        
        u_grad=[] 
        for i in range(d):
            u_grad.append(make_grad_i(i))
        return u_grad
                                                                   
                                                                        
    def alpha(x): 
        return torch.ones(x.size(0),1).to(device)
        # return 0.5 * torch.sin(6 * pi*x[:,0:1]) + 1. 

    def target(x):
        d = 3 
        cn =   7.03/d 
        z = torch.exp(-torch.sum( cn**2 * (x - 0.5)**2,dim = 1, keepdim = True)) 
        return z* ( -torch.sum(  (2 *cn**2 * (x - 0.5))**2 - 2*cn**2 ,dim = 1, keepdim = True) +1)

    def g_N(dim):
        def make_g(i):
            def g_i(x):
                d = 3 
                cn = 7.03 / d
                return torch.exp(-torch.sum(cn**2 * (x - 0.5)**2, dim=1, keepdim=True)) * (-2 * cn**2 * (x[:, i:i+1] - 0.5))
            return g_i

        bcs_N = []
        for i in range(dim):
            bcs_N.append((i, make_g(i)))
        
        return bcs_N


    function_name = "gaussian"
    filename_write = "data-revision/3DOGA-{}-order.txt".format(function_name)
    Nx = 50
    order = 3
    relu_k = 2
    f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
    f_write.write("RELU k = {}, Integration points: Nx {}, order {} \n".format(relu_k,Nx,order))
    f_write.close() 
    save = True 
    write2file = True
    rand_deter = 'rand'

    trial_num = 5 
    for N_list in [[2**3, 2**3, 2**3]]:
        for trial in range(trial_num):
            f_write = open(filename_write, "a")
            my_model = None
            exponent = 9
            num_epochs = 2**exponent
            plot_freq = num_epochs
            N = np.prod(N_list)
        
            err_QMC2, err_h10, my_model = OGANeumannReLU3D(my_model, alpha, target, g_N, u_exact, u_exact_grad, N_list, num_epochs, plot_freq, Nx=Nx, order=order, k=relu_k, rand_deter='rand', linear_solver="direct")

            if save:
                folder = 'data-revision/'
                filename = folder + 'errl2_OGA_3D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name, num_epochs, N, trial)
                torch.save(err_QMC2, filename)
                filename = folder + 'errh10_OGA_3D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name, num_epochs, N, trial)
                torch.save(err_h10, filename)
                filename = folder + 'model_OGA_3D_{}_neuron_{}_N_{}_randomized_trial_{}.pt'.format(function_name, num_epochs, N, trial)
                torch.save(my_model.state_dict(), filename)

            show_convergence_order2(err_QMC2, err_h10, exponent, N, filename_write, write2file=write2file)
            show_convergence_order_latex2(err_QMC2, err_h10, exponent, k=relu_k, d=dim)

if __name__ == "__main__":
    run_trials()

