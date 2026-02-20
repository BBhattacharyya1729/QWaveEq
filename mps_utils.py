from scipy.special import comb
from scipy.linalg import fractional_matrix_power as mpow
import numpy as np 
from functools import reduce 
from copy import deepcopy
from tqdm.notebook import tqdm
import pymanopt
from pymanopt.manifolds import Product, Stiefel


### Get the matrix product state on n sites for a polynomial with coefficients in the list a. reg is a list of 0's and 1's indicating what region of the interval is being used.### 
def MPS_poly(n,a, reg):
    p = len(a)-1
    t = ([1/2**i for i in range(1,n+1)])

    def phi(s,x):
        out = 0
        for k in range(s,p+1):
            out+=a[k] * comb(k,s) * x**(k-s)
        return out 

    M = [np.stack([np.array([phi(s,0) for s in range(p+1)]),
            np.array([phi(s,t[0]) for s in range(p+1)])])
    ]

    
    for k in range(1,n-1):
        M_temp = np.zeros((2,p+1,p+1))
        for i in range(p+1):
            for j in range(i+1):
                M_temp[0][i,j] = comb(i,i-j) *  0**(i-j) 
                M_temp[1][i,j] = comb(i,i-j) *  t[k]**(i-j)
        M.append(M_temp)    
        
    M.append(np.stack([np.concatenate([[1],np.zeros(p)]),np.array([t[-1]**i for i in range(p+1)])]))
    
    for i,v in enumerate(reg):
        M[i][1-v] *= 0 
    
    return M

### Direct sum of two matrices ###
def __block__(a,b):
    am,an = a.shape     
    bm,bn = b.shape
    return np.block( [[a, np.zeros((am,bn))], [np.zeros((bm,an)), b]])

### Sum of two MPS (corresponds to direct sum of components)###
def MPS_sum(M1,M2):
    M3 =  [np.concatenate([M1[0],M2[0]],axis=1)]
    for i in range(1,len(M1)-1):
        M3.append(
            np.array([__block__(M1[i][0],M2[i][0]),__block__(M1[i][1],M2[i][1])]
            )
        )
    M3.append(np.concatenate([M1[-1],M2[-1]],axis=1))
    return M3

### Cubic interpolation ##
def quad(x0,x1,f0,f1,df0,df1):
    A=np.array([[x0**3,x0**2,x0**1,1],
                [3*x0**2,2*x0,1,0],
                [x1**3,x1**2,x1**1,1],
                [3*x1**2,2*x1,1,0]])
    b = np.linalg.inv(A) @ np.array([f0,df0,f1,df1])
    return b[::-1]

### From a sequence of points, the function, and its derivative, get the relevant cubic polynomials between them ###
def get_polys(x,f,df):
    
    polys = [ ]
    for i in range(len(x)-1):
        polys.append(quad(x[i],x[i+1],f(x[i]),f(x[i+1]),df(x[i]),df(x[i+1])))
    
    return polys

### Apply a polynomial ###
def apply(p,x):
    return np.dot(p,[x**i for i in range(len(p))])

### Convert a Matrix product state to right cannonical form###
def right_cannonical_mps(M):
    cannon = deepcopy(M)
    n  = len(M)

    U,S,V = np.linalg.svd(cannon[-1].T,full_matrices=False)
    
    cannon[-1] =  V.T
    cannon[-2] = np.einsum('mkj,ji->mki',cannon[-2],U @np.diag(S)  )
    
    for i in range(1,n-1)[::-1]:
        ls,ms,ns = cannon[i].shape
        
        temp = np.permute_dims(cannon[i],(1,0,2))
        U,S,V = np.linalg.svd(np.reshape(temp,[ms, ls * ns]),full_matrices=False)
        V_temp = np.reshape(V, (V.shape[0],ls,V.shape[1]//ls))
        cannon[i] = np.permute_dims(V_temp,(1,0,2))

        if(i-1>0):
            cannon[i-1] = np.einsum('mkj,ji->mki',cannon[i-1],U @np.diag(S)  )
        else:
            cannon[i-1] = np.einsum('mj,ji->mi',cannon[i-1],U @ np.diag(S) )

    
    return cannon 

### Convert a matrix product state to left cannonical form ###
def left_cannonical_mps(M):
    cannon = deepcopy(M)
    n = len(M)
    U,S,V = np.linalg.svd(cannon[0],full_matrices=False)
    cannon[0] = U
    

    cannon[1] = np.einsum('ij,mjk->mik',np.diag(S) @ V,cannon[1])

    
    for i in range(1,n-1):
        ls,ms,ns = cannon[i].shape
        

        U,S,V = np.linalg.svd(np.reshape(cannon[i],[ls * ms,ns]),full_matrices=False)

        cannon[i] = np.reshape(U,[ls,U.shape[0]//ls,U.shape[1]])

        if(i+1<n-1):
            cannon[i+1] = np.einsum('ij,mjk->mik',np.diag(S) @ V,cannon[i+1])
        else:
            cannon[i+1] = np.einsum('ij,mj->mi',np.diag(S) @ V,cannon[i+1])

    return cannon 

### Truncate a matrix product state to a given order ###
def trunc_mps(M,order):
    cannon = deepcopy(M)
    
    cannon = right_cannonical_mps(cannon)
    
    n  = len(M)
    U,S,V = np.linalg.svd(cannon[0],full_matrices=False)
    U = U[:,0:order]
    S=S[0:order]
    V = V[0:order,:]

    cannon[0] = U
    cannon[1] = np.einsum('ij,mjk->mik',np.diag(S) @ V,cannon[1])
    
        
    for i in range(1,n-1):
        ls,ms,ns = cannon[i].shape

        U,S,V = np.linalg.svd(np.reshape(cannon[i],[ls * ms,ns]),full_matrices=False)
        U = U[:,0:order]
        S=S[0:order]
        V = V[0:order,:]
        cannon[i] = np.reshape(U,[ls,U.shape[0]//ls,U.shape[1]])

        if(i+1<n-1):
            cannon[i+1] = np.einsum('ij,mjk->mik',np.diag(S) @ V,cannon[i+1])
        else:
            cannon[i+1] = np.einsum('ij,mj->mi',np.diag(S) @ V,cannon[i+1])
    
    cannon[-1] = cannon[-1]/np.sqrt(cannon[-1][0].T.conjugate() @ cannon[-1][0] + cannon[-1][1].T.conjugate() @ cannon[-1][1]) 
    
    return cannon 

### Do the full contraction over MPS to get statevector. Notice that physical statevector indices are always first index in mps tensor###
def get_state(M):
    state = []
    n = len(M)
    for i in range(2**n):
        b = bin(i)[2:]
        b = '0' * (n - len(b)) + b
        b = [int(i) for i in b]
        M_b = [M[i][v] for i,v in enumerate(b)]
        state.append(reduce(lambda a,b:a@b, M_b))
    return np.array(state)

### Get layer from an EXACTLY bond order 2 left cannonical MPS ###
def get_layer(bond2):
    L = [bond2[0]]
    for i in range(1,len(bond2)-1):
        v = np.reshape(np.permute_dims(bond2[i],(1,0,2)),(4,2))
        L.append(((np.hstack([v,np.linalg.svd(v)[0][:,2:]]))))
    L.append(np.hstack([np.reshape(bond2[-1].T,(4,1)),np.linalg.svd(np.reshape(bond2[-1].T,(4,1)))[0][:,1:]]))
    return L[::-1]

### Get next layer from current sequence of layers ###
def get_next_layer(layers, target):
    disentangled_mps = deepcopy(target)
    for l in layers[::-1]:
        disentangled_mps = disentangle_layer(disentangled_mps, l)
    next_layer = get_layer(trunc_mps(disentangled_mps, 2))
    return [next_layer] + layers 

def zero_mps(n):
    return [np.array([[1,0],[0,0]])] + [np.array([[[1,0],[0,0]],[[0,0],[0,0]]])] * (n-2) + [np.array([[1,0],[0,0]])]

### Apply a single qubit unitary to a given mps site ###
def apply_single_unitary(U,mps,index):
    out = deepcopy(mps)
    if(index == 0):
        out[0] = U @ out[0]
    elif(index == len(out)-1):
        out[-1] = U @ out[-1]
    else:
        out[index] = np.einsum('ji,imk->jmk',U,out[index]) 
    return (out) 

### Apply a double unitary to a site (the indices are site and site+1).  Use SVD to decompose the new physical index pair. Notice that this increases bond dimension###
def apply_double_unitary(U,mps,index):
    out = deepcopy(mps)
    if(index == 0):
        res_U = np.reshape(U,(2,2,2,2))
        T_temp = np.einsum('klij,in,jnp->klp',res_U,out[0],out[1])

        ks,ls,ps = T_temp.shape
        rT_temp = np.reshape(T_temp,(ks,ls*ps))

        Us,S,Vs = np.linalg.svd(rT_temp,full_matrices=False)
        t1 = Us 
        t2 = np.permute_dims(np.reshape(np.diag(S) @ Vs,(ks,ls,ps)),(1,0,2))
        
        out[0] = t1
        out[1] = t2
        
    elif(index == len(out)-2):
        res_U = np.reshape(U,(2,2,2,2))
        T_temp = np.einsum('klij,imn,jn->klm',res_U,out[-2],out[-1])

        ks,ls,ms = T_temp.shape 
        rT_temp = np.permute_dims(T_temp,(0,2,1))
        rT_temp = np.reshape(rT_temp,(ks * ms,ls))
        Us,S,Vs = np.linalg.svd(rT_temp,full_matrices=False)

        t1 = np.reshape(Us,(ks,ms,ls))
        t2 = np.permute_dims(np.reshape(np.diag(S) @ Vs,(ls,ls)),(1,0))
        
        out[-2] = t1
        out[-1] = t2
        
    else:
        res_U = np.reshape(U,(2,2,2,2))
        T_temp = np.einsum('klij,imn,jnp->klmp',res_U,out[index],out[index+1])
        rT_temp = np.permute_dims(T_temp,(0,2,1,3))

        ks,ms,ls,ps = rT_temp.shape


        Us,S,Vs = np.linalg.svd(np.reshape(rT_temp,(ks * ms,ls * ps)),full_matrices=False)
        t1 = np.reshape(Us,(ks,ms,len(S)))
        t2 = np.permute_dims(np.reshape(np.diag(S) @ Vs,(len(S),ls,ps)),(1,0,2))
        
        out[index] = t1
        out[index+1] = t2

    return out 

### Contract two MPS's starting from the left and working towards the right ###
def left_mps_contract(mps1,mps2):
    extra = np.einsum('ij,ik->jk',mps1[0],mps2[0].conjugate())
    work1 = deepcopy(mps1[1:])
    work2 = [v.conjugate() for v in deepcopy(mps2[1:])]

    while(len(work1)>0 or len(work2)>0):

        if(len(work1) == len(work2)):
            assert len(extra.shape) == 2
            if(len(work2[0].shape) == 3):
                extra = np.einsum('ij,kjl->ikl',extra,work2[0])
            else:
                extra = np.einsum('ij,kj->ik',extra,work2[0])
            work2 = work2[1:]
        else:
            assert (len(work1) - len(work2)) == 1
            if(len(work1[0].shape) == 3):
                assert len(extra.shape) == 3
                extra = np.einsum('ikl,kij->jl',extra,work1[0])
            else:
                assert len(extra.shape) == 2
                extra = np.einsum('ik,ki->',extra,work1[0])
            work1 = work1[1:]

    return extra 

### Contract two MPS's starting from the right and working towards the left ###
def right_mps_contract(mps1,mps2):
    extra = np.einsum('ij,ik->jk',mps1[-1],mps2[-1].conjugate())
    work1 = deepcopy(mps1[:-1])
    work2 = [v.conjugate() for v in deepcopy(mps2[:-1])]
    
    while(len(work1)>0 or len(work2)>0):

        if(len(work1) == len(work2)):
            assert len(extra.shape) == 2
            if(len(work2[-1].shape) == 3):
                extra = np.einsum('ij,klj->ikl',extra,work2[-1])
            else:
                extra = np.einsum('ij,kj->ik',extra,work2[-1])
            work2 = work2[:-1]
        else:
            assert (len(work1) - len(work2)) == 1
            if(len(work1[-1].shape) == 3):
                assert len(extra.shape) == 3
                extra = np.einsum('ikl,kji->jl',extra,work1[-1])
            else:
                assert len(extra.shape) == 2
                extra = np.einsum('ik,ki->',extra,work1[-1])
            work1 = work1[:-1]

    return extra 


def entangle_layer(init_state, layer):
    n = len(init_state)
    out = deepcopy(init_state)
    for i,U in enumerate(layer[:-1]):
        out = apply_double_unitary(U, out, n - i - 2)
    out = apply_single_unitary(layer[-1], out, 0)
    return out

def disentangle_layer(init_state, layer):
    out = deepcopy(init_state)
    out = apply_single_unitary(layer[-1].T.conjugate(), out, 0)
    for i,U in enumerate(layer[:-1][::-1]):
        out = apply_double_unitary(U.T.conjugate(), out, i)
    return out

def get_double_fidelity_op(mps1,mps2,index):
    
    if(index == 0):
        return __get_left_double_fidelity_op__(mps1,mps2)
    elif(index == len(mps1) - 2):
        return __get_right_double_fidelity_op__(mps1,mps2)
    
    extra1 = left_mps_contract(mps1[:index],mps2[:index])
    extra2 = right_mps_contract(mps1[index+2:],mps2[index+2:])

    
    middle1 = mps1[index:index+2]
    middle2 = [v.conjugate() for v in mps2[index:index+2]]

        
    out = np.einsum('ij,kjl->ikl',extra1,middle2[0])
    out = np.einsum('ikl,mip->klmp',out,middle1[0])
    out = np.einsum('klmp,ulv->kmpuv',out,middle2[1])
    out = np.einsum('kmpuv,jpl->kuvlmj',out,middle1[1])
    out = np.einsum('kuvlmj,lv->mjku',out,extra2)
    
    return out.reshape(4,4)

def __get_left_double_fidelity_op__(mps1,mps2):
    extra2 = right_mps_contract(mps1[2:],mps2[2:])

    
    middle1 = mps1[0:2]
    middle2 = [v.conjugate() for v in mps2[0:2]]
    
    out = np.einsum('ij,kjl->ikl',middle1[0],middle1[1])
    out = np.einsum('ikl,lv->ikv',out,extra2)
    out = np.einsum('ikv,umv->ikmu',out,middle2[1])
    out = np.einsum('ikmu,jm->ikju',out,middle2[0])
    
    return out.reshape(4,4)

def __get_right_double_fidelity_op__(mps1,mps2):
        
    extra1 = left_mps_contract(mps1[:len(mps1)-2],mps2[:len(mps2)-2])

    
    middle1 = mps1[len(mps1)-2:len(mps1)]
    middle2 = [v.conjugate() for v in mps2[len(mps2)-2:len(mps2)]]
    
    out = np.einsum('ij,kil->klj',extra1,middle1[0])
    out = np.einsum('klj,ml->kmj',out,middle1[1])
    out = np.einsum('kmj,pjl->kmpl',out,middle2[0])
    out = np.einsum('kmpl,nl->kmpn',out,middle2[1])
    
    
    return out.reshape(4,4)

def get_single_fidelity_op(mps1,mps2):
    extra1 = right_mps_contract(mps1[1:],mps2[1:])
    
    out = np.einsum('ij,ki->kj',extra1,mps1[0])
    out = np.einsum('kj,mj->km',out,mps2[0].conjugate())
    

    return out


# --------------------------------------------------
# Helper: apply a gate (optionally daggered)
# --------------------------------------------------
def apply_gate(g, state, idx, dagger=False):
    if dagger:
        g = g.conjugate().T
    if len(g) == 2:
        return apply_single_unitary(g, state, idx)
    else:
        return apply_double_unitary(g, state, idx)


# --------------------------------------------------
# Efficient computation of all F operators
# --------------------------------------------------
def compute_F_list(gates, indices, init, targ):
    G = len(gates)

    # Prefix states: R[i] = U_{i-1} ... U_0 |init>
    R = [None] * (G + 1)
    R[0] = init
    for i in range(G):
        R[i + 1] = apply_gate(gates[i], R[i], indices[i])

    # Suffix states: L[i] = U_{i+1}^† ... U_{G-1}^† |targ>
    L = [None] * (G + 1)
    L[G] = targ
    for i in reversed(range(G)):
        L[i] = apply_gate(gates[i], L[i + 1], indices[i], dagger=True)

    # Fidelity operators
    F_list = []
    for i in range(G):
        if len(gates[i]) == 2:
            F_list.append(get_single_fidelity_op(R[i], L[i + 1]))
        else:
            F_list.append(
                get_double_fidelity_op(R[i], L[i + 1], indices[i])
            )

    return F_list


# --------------------------------------------------
# Gate update
# --------------------------------------------------
def update_gate(g, F, r):
    U, S, Vh = np.linalg.svd(F)
    F_op = Vh.conjugate().T @ U.conjugate().T

    new_op = g @ mpow(g.conjugate().T @ F_op, r)

    U, S, Vh = np.linalg.svd(new_op)
    return U @ Vh


# --------------------------------------------------
# One optimization step over all gates
# --------------------------------------------------
def step_gates(gates, indices, init, targ, r):
    F_list = compute_F_list(gates, indices, init, targ)
    for i in range(len(gates)):
        gates[i] = update_gate(gates[i], F_list[i], r)
    return gates


# --------------------------------------------------
# Main optimization routine
# --------------------------------------------------
def optimize_mps_circuit(targ, num_layers, r=1e-3, opt_iters=1000):
    layers = []
    gates = []
    indices = []

    layers_hist = []   # history after full layer updates
    gates_hist = []    # full gate update history

    n = len(targ)
    init = zero_mps(n)

    for _ in range(num_layers):
        # Build next layer
        layers = get_next_layer(layers, targ)
        gates = sum(layers, [])
        indices += list(range(n - 1))[::-1] + [0]

        # Optimize gates
        for _ in range(opt_iters):
            gates = step_gates(gates, indices, init, targ, r)
            gates_hist.append(deepcopy(gates))

        # Re-pack gates into layers
        layers = [gates[i:i + n] for i in range(0, len(gates), n)]
        layers_hist.append(deepcopy(layers))

        print("----------------------------------------------")

    return layers_hist, gates_hist

# def manifold_gate_opt(targ_mps,init_gates):
#     n = len(targ_mps)
#     psi_init = zero_mps(n)
#     num_layers = len(init_gates)//n 

#     idx_list = (list(range(0,n-1))[::-1] + [0]) * num_layers
#     shapes = ([(4,4)] * (n-1) + [(2,2)]) * num_layers
    
#     def overlap(gates):
#         layers = [gates[i * n:(i+1) * n] for i in range(num_layers)]
#         eval_state = deepcopy(psi_init)
#         for l in layers:
#             eval_state = entangle_layer(eval_state, l)
        
#         return left_mps_contract(targ_mps, eval_state)

#     def overlap_grad(gates):
#         grads = []
#         for i in range(len(gates)):
#             left_mps = deepcopy(targ_mps)
#             right_mps = deepcopy(psi_init)
            
#             prev_op, prev_idx = gates[:i], idx_list[:i]
#             after_op, after_idx = gates[(i+1):], idx_list[(i+1):]
            
#             for (U, _) in zip(prev_op, prev_idx):
#                 if(len(U) == 2):
#                     right_mps = apply_single_unitary(U, right_mps, _)
#                 else:
#                     right_mps = apply_double_unitary(U, right_mps, _)
                    
#             for (U, _) in zip(after_op[::-1], after_idx[::-1]):
#                 if(len(U) == 2):
#                     left_mps = apply_single_unitary(U.T, left_mps, _)
#                 else:
#                     left_mps = apply_double_unitary(U.T, left_mps, _)
            
#             if(len(gates[i]) == 2):
#                 grads.append(get_single_fidelity_op(left_mps, right_mps))
#             else:
#                 grads.append(get_double_fidelity_op(left_mps, right_mps, idx_list[i]))
#         return grads


#     manifold = Product([Stiefel(*s) for s in shapes])
#     hist = []
#     @pymanopt.function.numpy(manifold)
#     def log_fidelity(*gates):
#         F = overlap(gates)
#         hist.append(gates)
#         return np.log(1-F**2)

#     @pymanopt.function.numpy(manifold)
#     def log_fidelity_grad(*gates):
#         F = overlap(gates)
#         k = -(2 * F)/(1 - F**2)
#         G = overlap_grad(gates)
#         return [k * g for g in G]

#     problem = pymanopt.Problem(manifold, log_fidelity, euclidean_gradient=log_fidelity_grad)
#     optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations = 5000, max_time = 10000)
#     result = optimizer.run(problem, initial_point=init_gates)
    
#     return result, hist

# def optimize_mps_circuit(targ_mps, num_layers):
#     layers = []
#     full_hist = []
#     results = []
#     n = len(targ_mps)
#     for i in range(num_layers):
#         print("==========================================")
#         print("Optimization for Layer ", i+1)
#         print("==========================================")
#         layers = get_next_layer(layers, targ_mps)
#         result, hist = manifold_gate_opt(targ_mps,sum(layers, []))
#         full_hist.append(hist)
#         results.append(result)
#         layers = [result.point[i * n:(i+1) * n] for i in range(len(result.point)//n)]
#     return layers, results, full_hist


# def entangle_layer__(mps,L):
#     out = [np.array(m,dtype=complex)  for m in deepcopy(mps)]
#     out = deepcopy(mps)
#     for (U,idx) in zip(L[:-1],range(0,len(L)-1)[::-1]):
#         out = apply_double_unitary(U,out,idx)
#     return right_cannonical_mps(apply_single_unitary(L[-1],out,0))

# def disentangle_layer__(mps,L):
#     out = [np.array(m,dtype=complex)  for m in deepcopy(mps)]
#     out = apply_single_unitary(L[-1].T.conjugate(),out,0)
#     for (U,idx) in zip(L[:-1][::-1],range(0,len(L)-1)):
#         out = apply_double_unitary(U.T.conjugate(),out,idx)
#     return right_cannonical_mps(out)


### Qiskit methods 
# import qiskit 
# d2ZYZ = qiskit.synthesis.TwoQubitBasisDecomposer(qiskit.circuit.library.CXGate(), euler_basis="ZYZ")
# d1ZYZ = qiskit.synthesis.OneQubitEulerDecomposer(basis = "ZYZ")

# def nearest_unitary(M):
#     U,_,V = np.linalg.svd(M)
#     return U @ V

# def circ_from_layer(L):
    

#     circs = [d2ZYZ(np.reshape(np.permute_dims(np.reshape(nearest_unitary(U),(2,2,2,2)),(1,0,3,2)),(4,4))) for U in L[:-1]] + [d1ZYZ(nearest_unitary(L[-1]))] 
#     circ = qiskit.QuantumCircuit(len(L))
    
#     for (c,idx) in zip(circs[:-1],range(0,len(L)-1)[::-1]):
#         circ = circ.compose(c,[idx,idx+1])
#     circ = circ.compose(circs[-1],[0])
#     return circ

# def circ_from_layers(L_list):
#     circ = circ_from_layer(L_list[0])
#     for L in L_list[1:]:
#         circ = circ.compose(circ_from_layer(L),range(len(L)))
#     return circ

# def get_layers(mps,L):
#     layers = []
#     mps_inter = []
#     for i in range(L):
#         if(i==0):
#             new_layer = get_layer(trunc_mps(mps,2))
#             new_mps = disentangle_layer__(mps,new_layer)
    
#             layers.append(new_layer)
#             mps_inter.append(new_mps)
#         else:
#             new_layer = get_layer(trunc_mps(mps_inter[-1],2))
#             new_mps = disentangle_layer__(mps_inter[-1],new_layer)

#             layers.append(new_layer)
#             mps_inter.append(new_mps)
#     return (layers[::-1])


# """
# Given two states mps2 and mps1 and a list of gates (real) U_1,U_2,...,U_n
# Compute the gradient of 
# <mps2|U_n...U_i...U_2U_1|mps1>
# With respect to the real components of U_i
# """
# def gate_inner_gradient(gates,indices,mps1,mps2,gate_idx):
#     gates1 = gates[:gate_idx]
#     gates2 = gates[gate_idx+1:]
    
#     idx1 = indices[:gate_idx]
#     idx2 = indices[gate_idx+1:]

#     new_mps1 = deepcopy(mps1)
#     new_mps2 = deepcopy(mps2)
    
    
#     for (idx,g) in zip(idx1,gates1):
#         if(len(g) == 4):
#             new_mps1 = apply_double_unitary(g,new_mps1,idx)
#         else:
#             new_mps1 = apply_single_unitary(g,new_mps1,0)
            
#     for (idx,g) in zip(idx2[::-1],gates2[::-1]):
#         if(len(g) == 4):
#             new_mps2 = apply_double_unitary(g.T.conjugate(),new_mps2,idx)
#         else:
#             new_mps2 = apply_single_unitary(g.T.conjugate(),new_mps2,0)
    
#     if(len(gates[gate_idx])==4):
#         eps = left_mps_contract(apply_double_unitary(gates[gate_idx],new_mps1,indices[gate_idx]),new_mps2)
#         F = get_double_fidelity_op(new_mps2,new_mps1,indices[gate_idx]).conjugate()
#         G = 1j * F
#     else:
#         eps = left_mps_contract(apply_single_unitary(gates[gate_idx],new_mps1,0),new_mps2)
#         F = get_single_fidelity_op(new_mps2,new_mps1).conjugate()

#     return F,eps

# """
# Given two states mps2 and mps1 and a list of gates (real) U_1,U_2,...,U_n
# Compute the gradient of 
# 1-|<mps2|U_n...U_i...U_2U_1|mps1>|^2
# With respect to the real components of U_i
# """
# def gate_infidelity_gradient(gates,indices,mps1,mps2,gate_idx):
#     F,eps = gate_inner_gradient(gates,indices,mps1,mps2,gate_idx)
#     F_full  = -2 * F * eps
#     return F_full, eps


# """
# Compute the retraction of a matrix A
# """
# def retract(A):
#     U,_,V = np.linalg.svd(A)
#     return U @ V 

# """
# Compute the projection of a tangent v given a unitary G
# """
# def project(v,G):
#     return v - 1/2 * G @ (v.conjugate().T @ G + G.conjugate().T @ v)

# """
# Project onto G after moving along v
# """
# def transport(G,v,w):
#     return project(w,retract(G+v))


# """
# Riemannian Gradient Descent  
# """
# def gradient_descent(initial,target,gates,indices,alpha,reps):
#     new_gates=deepcopy(gates)
#     infid_hist = []
#     for i in tqdm(range(reps)):
#         for gate_idx in range(len(gates)):
#             F_full, eps = gate_infidelity_gradient(new_gates,indices,initial,target,gate_idx)
#             U = new_gates[gate_idx]
            
#             v = F_full
#             proj_v = project(v,U)
            
#             new_gates[gate_idx] = retract(U - alpha * proj_v)
            
#         infid_hist.append(1-abs(eps)**2)
            
#     return new_gates,infid_hist 

# """
# # """
# def ADAM_decent(initial,target,gates,indices,alpha,reps,beta1=0.9,beta2=0.999):
#     transp_last  = [np.zeros(g.shape) for g in gates]
#     v_last = [0 for g in gates]
    
#     infid_hist = []
#     new_gates=deepcopy(gates)
    
#     for i in tqdm(range(reps)):
#         for gate_idx in range(len(gates)):
#             F_full, eps = gate_infidelity_gradient(new_gates,indices,initial,target,gate_idx)
#             U = new_gates[gate_idx]
            
#             vec = F_full
#             proj_v = project(vec,U)
            
#             m = beta1 * project(transp_last[gate_idx],U)+ (1-beta1) * proj_v
#             v = beta2 * v_last[gate_idx] + (1-beta2) * np.trace(proj_v.conjugate().T @ proj_v)
            
#             step = -alpha * (m)/(np.sqrt(v)+1e-8)
            
#             new_gates[gate_idx] = retract(U + step)
            
#             transp_last[gate_idx] = project(m,new_gates[gate_idx]) 
#             v_last[gate_idx] = np.max([v,v_last[gate_idx]]) 
    
#         infid_hist.append(1-abs(eps)**2)
        
#     return new_gates,infid_hist 

# """
# Layers <-> gate/index pairs 
# """
# def layers_convert(L_list):
#     gates = []
#     indices = []
#     for L in L_list:
#         for (n,g) in enumerate(L):
#             if(len(g) == 2):
#                 gates.append(g)
#                 indices.append(0)
#             else: 
#                 gates.append(g)
#                 indices.append(len(L)-n-2)
#     return gates,indices

# def gates_convert(gates,indices,n):
#     return [gates[i:i + n] for i in range(0, len(gates), n)]
                


# def optimize_layers(k,mps,reps,alpha=1e-3,beta1=0.9,beta2=0.999):

#     grad_layers = []
#     grad_opts = []
#     layer_hist = []
#     n = len(mps)
#     psi_init = [np.array([[1,0],[0,0]])] + [np.array([[[1,0],[0,0]],[[0,0],[0,0]]])] * (n-2) + [np.array([[1,0],[0,0]])]

#     for i in range(k):
#         if(i==0):
#             new_layer = get_layer(trunc_mps(mps,2))
#             grad_layers.append(new_layer)
            
#             gates,indices = layers_convert(grad_layers[::-1])
            
#             new_gates,eps = ADAM_decent(psi_init,mps,gates,indices,alpha,reps,beta1=beta1,beta2=beta2)
            
#             new_L = gates_convert(new_gates,indices,n)
            
#             grad_layers = new_L[::-1]
#             layer_hist.append(deepcopy(grad_layers[::-1]))
#             grad_opts.append(eps)
#         else:
#             current_mps = deepcopy(mps)
#             for i in grad_layers:
#                 current_mps = disentangle_layer__(current_mps,i)
            
#             new_layer = get_layer(trunc_mps(current_mps,2))
#             grad_layers.append(new_layer)

#             gates,indices = layers_convert(grad_layers[::-1])

#             new_gates,eps = ADAM_decent(psi_init,mps,gates,indices,alpha,reps,beta1=beta1,beta2=beta2)
            
#             new_L = gates_convert(new_gates,indices,n)
            
#             grad_layers = new_L[::-1]
#             layer_hist.append(deepcopy(grad_layers[::-1]))
#             grad_opts.append(eps)
            
#     grad_layers = grad_layers[::-1]

#     return grad_layers, layer_hist, grad_opts



# from scipy.linalg import fractional_matrix_power
# def layer_update(mps1,mps2,L,r):
#     new_L = deepcopy(L)
#     n = len(mps1)
    
#     for gate_idx in range(0,n):
#         gates1 = new_L[:gate_idx]
#         gates2 = new_L[gate_idx+1:]
                
#         new_mps1 = deepcopy(mps1)
#         new_mps2 = deepcopy(mps2)


#         for (idx,g) in enumerate(gates1):
#             if(len(g) == 4):
#                 new_mps1 = apply_double_unitary(g,new_mps1, n - 2 -idx)
#             else:
#                 new_mps1 = apply_single_unitary(g,new_mps1,0)
            
#         for (idx, g) in enumerate(gates2[::-1]):
#             if(len(g) == 4):
#                 new_mps2 = apply_double_unitary(g.T.conjugate(),new_mps2,idx-1)
#             else:
#                 new_mps2 = apply_single_unitary(g.T.conjugate(),new_mps2,0)
        
#         if(gate_idx == n-1):
#             F = get_single_fidelity_op(new_mps2,new_mps1).conjugate()
#         else:
#             F = get_double_fidelity_op(new_mps2,new_mps1,n-2-gate_idx).conjugate()
#         U,_,V = np.linalg.svd(F)
#         M = (U @ V)
        
        
#         new_L[gate_idx] = new_L[gate_idx] @ fractional_matrix_power(new_L[gate_idx].conjugate().T @ M, r)
        
#     return new_L
    
# def layer_grad_optimize(initial,target,initial_L, r=1e-2, iters=500):
#     new_L = deepcopy(initial_L)
    
#     data = [abs(left_mps_contract(target,(entangle_layer__(initial,new_L))))**2]
#     L_list = [deepcopy(new_L)]
    
#     for i in range(iters):
#         new_L = layer_update(initial,target,new_L ,r)
#         data.append(abs(left_mps_contract(target,(entangle_layer__(initial,new_L))))**2)
#         L_list.append(deepcopy(new_L))
        
#     return np.max(data),L_list[np.argmax(data)], (data,L_list)

# # def multi_layer_grad_optimize(target,L_list, r=1e-2, iters=10):
#     initial = [np.array([[1,0],[0,0]])] + [np.array([[[1,0],[0,0]],[[0,0],[0,0]]])] * (len(target)-2) + [np.array([[1,0],[0,0]])]
    
#     new_L_List = deepcopy(L_list)
#     hist = []
#     hists_full = []
#     for _ in range(iters):
#         for (idx,Lo) in enumerate(new_L_List):
            
#             pre_layers = new_L_List[:idx]
#             post_layers = new_L_List[idx+1:]
            
#             current_initial = deepcopy(initial)
#             for L in pre_layers:
#                 current_initial = entangle_layer__(current_initial,L)
            
#             current_target = deepcopy(target)
#             for L in post_layers[::-1]:
#                 current_target = disentangle_layer__(current_target,L)
            
#             v,L,h = layer_grad_optimize(current_initial,current_target,Lo, r=r, iters=200)
#             new_L_List[idx] = L
#             hist.append(v)
#             hists_full.append(h)
#             #print(h[0][0],h[0][-1])
#         print(_)
#     return new_L_List,hist,hists_full
        
        
        
