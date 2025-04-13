from scipy.special import comb
import numpy as np 
from functools import reduce 
from copy import deepcopy

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

def __block__(a,b):
    am,an = a.shape     
    bm,bn = b.shape
    return np.block( [[a, np.zeros((am,bn))], [np.zeros((bm,an)), b]])

def MPS_sum(M1,M2):
    M3 =  [np.concatenate([M1[0],M2[0]],axis=1)]
    for i in range(1,len(M1)-1):
        M3.append(
            np.array([__block__(M1[i][0],M2[i][0]),__block__(M1[i][1],M2[i][1])]
            )
        )
    M3.append(np.concatenate([M1[-1],M2[-1]],axis=1))
    return M3

def quad(x0,x1,f0,f1,df0,df1):
    A=np.array([[x0**3,x0**2,x0**1,1],
                [3*x0**2,2*x0,1,0],
                [x1**3,x1**2,x1**1,1],
                [3*x1**2,2*x1,1,0]])
    b = np.linalg.inv(A) @ np.array([f0,df0,f1,df1])
    return b[::-1]

def get_polys(x,f,df):
    
    polys = [ ]
    for i in range(len(x)-1):
        polys.append(quad(x[i],x[i+1],f(x[i]),f(x[i+1]),df(x[i]),df(x[i+1])))
    
    return polys

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
    
    cannon[-1] = cannon[-1]/np.sqrt(cannon[-1][0].T @ cannon[-1][0] + cannon[-1][1].T @ cannon[-1][1]) 
    
    return cannon 

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

def get_layer(bond2):
    L = [bond2[0]]
    for i in range(1,len(bond2)-1):
        v = np.reshape(np.permute_dims(bond2[i],(1,0,2)),(4,2))
        L.append(((np.hstack([v,np.linalg.svd(v)[0][:,2:]]))))
    L.append(np.hstack([np.reshape(bond2[-1].T,(4,1)),np.linalg.svd(np.reshape(bond2[-1].T,(4,1)))[0][:,1:]]))
    return L[::-1]

def __kron__(l):
    return reduce(lambda a,b:np.kron(a,b), l)


def apply_single_unitary(U,mps,index):
    out = deepcopy(mps)
    if(index == 0):
        out[0] = U @ out[0]
    elif(index == len(out)-1):
        out[-1] = U @ out[-1]
    else:
        out[index] = np.einsum('ji,imk->jmk',U,out[index]) 
    return (out) 

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

    return right_cannonical_mps(out)



def entangle_layer__(mps,L):
    out = deepcopy(mps)
    for (U,idx) in zip(L[:-1],range(0,len(L)-1)[::-1]):
        out = apply_double_unitary(U,out,idx)
    return apply_single_unitary(L[-1],out,0)

def disentangle_layer__(mps,L):
    out = deepcopy(mps)
    out = apply_single_unitary(L[-1].T,out,0)
    for (U,idx) in zip(L[:-1][::-1],range(0,len(L)-1)):
        out = apply_double_unitary(U.T,out,idx)
    return out


def check_left(mps):
    assert np.all(np.isclose(mps[0] @ mps[0].T, np.eye(mps[0].shape[0])))
    for i in range(1,len(mps)-1):
        assert np.all(np.isclose(np.sum([j.T @ j for j in mps[i]],axis=0),np.eye(mps[i][0].shape[1])))
        
    assert (np.isclose(np.sqrt(mps[-1][0].T @ mps[-1][0] + mps[-1][1].T @ mps[-1][1]),1))


### Qiskit methods 
import qiskit 
d2ZYZ = qiskit.synthesis.TwoQubitBasisDecomposer(qiskit.circuit.library.CXGate(), euler_basis="ZYZ")
d1ZYZ = qiskit.synthesis.OneQubitEulerDecomposer(basis = "ZYZ")

def circ_from_layer(L):
    circs = [d2ZYZ(np.reshape(np.permute_dims(np.reshape(U,(2,2,2,2)),(1,0,3,2)),(4,4))) for U in L[:-1]] + [d1ZYZ(L[-1])] 
    circ = qiskit.QuantumCircuit(len(L))
    
    for (c,idx) in zip(circs[:-1],range(0,len(L)-1)[::-1]):
        circ.append(c,[idx,idx+1])
    circ.append(circs[-1],[0])
    return circ

def circ_from_layers(L_list):
    circ = circ_from_layer(L_list[0])
    for L in L_list[1:]:
        circ.append(circ_from_layer(L),range(len(L)))
    return circ

def circ_from_mps(mps,L):
    layers = []
    mps_inter = []
    for i in range(L):
        if(i==0):
            new_layer = get_layer(trunc_mps(mps,2))
            new_mps = disentangle_layer__(mps,new_layer)

            layers.append(new_layer)
            mps_inter.append(new_mps)
        else:
            new_layer = get_layer(trunc_mps(mps_inter[-1],2))
            new_mps = disentangle_layer__(mps_inter[-1],new_layer)

            layers.append(new_layer)
            mps_inter.append(new_mps)
    return circ_from_layers(layers[::-1])