from greedy_general_elliptic_utils import * 


m1 = 2
m2 = 2
m3 = 2 
def u_exact(x):
    return torch.cos(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])  

def u_exact_grad():

    def grad_1(x):
        return - m1*pi* torch.sin(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])   
    def grad_2(x):
        return - m2*pi* torch.cos(m1*pi*x[:,0:1])*torch.sin( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])  
    def grad_3(x):
        return - m3*pi* torch.cos(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.sin(m3*pi*x[:,2:3])   
    
    u_grad=[grad_1, grad_2,grad_3] 

    return u_grad

def laplace_u_exact(x):
    return -((m1*pi)**2 + (m2*pi)**2 +(m3*pi)**2 )  * torch.cos(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])

def convection_term(x):
    grad1 = - m1*pi* torch.sin(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])   
    grad2 = - m2*pi* torch.cos(m1*pi*x[:,0:1])*torch.sin( m2*pi*x[:,1:2]) * torch.cos(m3*pi*x[:,2:3])  
    grad3 = - m3*pi* torch.cos(m1*pi*x[:,0:1])*torch.cos( m2*pi*x[:,1:2]) * torch.sin(m3*pi*x[:,2:3])    
    return BETA * grad1 + BETA * grad2 + BETA * grad3  

def rhs(x):
    return  -laplace_u_exact(x) + convection_term(x) + LAMBDA * u_exact(x)  

g_N = None 

function_name = "cosine"
Nx = 50 
order = 2
exponent = 9
num_epochs = 2**exponent  
plot_freq = num_epochs 
rand_deter = 'rand'
memory = 2**29
activation = 'tanh' 
relu_k = 3 # not used if activation != relu 
folder = 'data-revision1/'
filename_write = folder + "3DOGA-{}-{}-general-elliptic-a{}-b{}-c{}-order.txt".format(activation,function_name,1,BETA,LAMBDA)
f_write = open(filename_write, "a" if os.path.exists(filename_write) else "w")
f_write.write("{}, Integration points: Nx {}, order {} \n".format(activation, relu_k,Nx,order))
f_write.close() 

save = True 
write2file = True

trial_num = 5 
for N_list in [[2**3,2**3,2**3]]: # ,[2**6,2**6],[2**7,2**7] 
    for trial in range(trial_num): 
        f_write = open(filename_write, "a")
        my_model = None 
        N = np.prod(N_list)
        err_QMC2, err_h10, my_model = OGAGeneralEllipticReLUNDim(my_model,rhs, u_exact, u_exact_grad,g_N, N_list,num_epochs,plot_freq, Nx, order, activation= activation, k = relu_k, rand_deter = rand_deter, solver = "direct",memory = memory)
        if save: 
        
            filename = folder + 'errl2_OGA_3D_{}_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,activation, num_epochs,N,trial)
            torch.save(err_QMC2,filename) 
            filename = folder + 'err_h10_OGA_3D_{}_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,activation, num_epochs,N,trial)
            torch.save(err_h10,filename)
            filename = folder + 'model_OGA_3D_{}_{}_neuron_{}_N_{}_rand_trial_{}.pt'.format(function_name,activation, num_epochs,N,trial)
            torch.save(my_model.state_dict(),filename)
        show_convergence_order2(err_QMC2,err_h10,exponent,N,relu_k, DIMENSION, filename_write,write2file = write2file)
        show_convergence_order_latex2(err_QMC2,err_h10,exponent,k =relu_k,d = DIMENSION)

