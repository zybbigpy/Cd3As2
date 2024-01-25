import csv
import numpy as np
from numba import njit, complex64
import matplotlib.pyplot as plt


fname = "/Users/wmiao/Desktop/test_D.csv"
num_band = 16
theta = np.pi/2
Bx  = 10

def set_tb_disp_kmesh(n_k: int, high_symm_pnts: dict) -> tuple:

    num_sec = len(high_symm_pnts)
    num_kpt = n_k*(num_sec-1)
    length = 0

    klen = np.zeros((num_sec), float)
    kline = np.zeros((num_kpt+1), float)
    kmesh = np.zeros((num_kpt+1, 2), float)
    ksec = []
    
    for key in high_symm_pnts:
        ksec.append(high_symm_pnts[key])

    for i in range(num_sec-1):
        vec = ksec[i+1]-ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i+1] = klen[i]+length

        for ikpt in range(n_k):
            kline[ikpt+i*n_k] = klen[i]+ikpt*length/n_k
            kmesh[ikpt+i*n_k] = ksec[i]+ikpt*vec/n_k
    kline[num_kpt] = kline[(num_sec-2)*n_k]+length
    kmesh[num_kpt] = ksec[-1]

    return (kline, kmesh)


def rot_mat(theta):

    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rt_mtrx


with open(fname, "r") as file:
    reader = csv.reader(file)
    data = []
    for row in reader:
        line_data = []
        for x in row:
            s = x.replace("*I", "j")
            line_data.append(s.replace("*^", "e"))
        data.append(line_data)


print("finish loading data of coeff of kp.")
largeComplexArray = np.array(data, dtype=complex)
largeComplexArray = largeComplexArray.reshape(15, 200, 200)

term_const = largeComplexArray[0, :, :]
term_k1    = largeComplexArray[1, :, :]
term_k2    = largeComplexArray[2, :, :]
term_k1k1  = largeComplexArray[3, :, :]
term_k1k2  = largeComplexArray[4, :, :]
term_k2k1  = largeComplexArray[5, :, :]
term_k2k2  = largeComplexArray[6, :, :]

term_k1k1k1 = largeComplexArray[7,:,:]
term_k1k1k2 = largeComplexArray[8,:,:]
term_k1k2k1 = largeComplexArray[9,:,:]
term_k1k2k2 = largeComplexArray[10,:,:]
term_k2k1k1 = largeComplexArray[11,:,:]
term_k2k1k2 = largeComplexArray[12,:,:]
term_k2k2k1 = largeComplexArray[13,:,:]
term_k2k2k2 = largeComplexArray[14,:,:]

@njit
def cal_hamk(k1, k2):

    hamk =  term_const+k1*term_k1+k2*term_k2+k1*k2*term_k1k2+k1*k2*term_k2k1+k1**2*term_k1k1+term_k2k2*k2**2 \
           +term_k1k1k1*k1*k1*k1+term_k1k1k2*k1*k1*k2+term_k1k2k1*k1*k2*k1+term_k1k2k2*k1*k2*k2 \
           +term_k2k1k1*k2*k1*k1+term_k2k1k2*k2*k1*k2+term_k2k2k1*k2*k2*k1+term_k2k2k2*k2*k2*k2

    return hamk

@njit
def make_vkx(k1, k2):

    vkx = term_k1+term_k1k2*k2+term_k2k1*k2+term_k1k1*2*k1 \
        + term_k1k1k1*3*k1*k1+term_k1k1k2*2*k1*k2+term_k1k2k1*2*k1*k2+term_k1k2k2*k2*k2\
        + term_k2k1k1*2*k1*k2+term_k2k1k2*k2*k2+term_k2k2k1*k2*k2

    return vkx[:num_band, :num_band]


@njit
def make_vky(k1, k2):

    vky = term_k2+term_k1k2*k1+term_k2k1*k1+term_k2k2*2*k2\
        + term_k1k1k2*k1*k1+term_k1k2k1*k1*k1+term_k1k2k2*k1*k2*2 \
        +term_k2k1k1*k1*k1+term_k2k1k2*k1*k2*2+term_k2k2k1*k2*k1*2+term_k2k2k2*k2*k2*3

    return vky[:num_band, :num_band]


@njit
def make_Zeeman(B, theta=theta):
    
    muB = 5.7884*1e-5 # [eV/Tesla]
    g1  = 11*muB
    g2  = 0.7*muB  
    gz1 = 10*muB
    gz2 = -5*muB

    Bx = B*np.cos(theta)
    By = B*np.sin(theta)
    B1 = (1/(3*np.sqrt(2))+np.sqrt(2)/3)*Bx+1/np.sqrt(3)*By
    B2 = (-1/(3*np.sqrt(2))-np.sqrt(2)/3)*Bx+1/np.sqrt(3)*By
    B3 = -1/np.sqrt(3)*By

    print("B1 B2 B3 in [T]:", B1, B2, B3)
    assert np.allclose(B1+B2+2*B3, 0) == True
    assert np.allclose(np.sqrt(B1**2+B2**2+B3**2), B) == True 

    zeeman = np.zeros((200, 200), dtype=complex64)
    for i in range(50):
        zeeman[4*i:4*i+4, 4*i:4*i+4] = np.array([[gz1*B3, g1*(B1-1j*B2), 0, 0],
                                                 [g1*(B1+1j*B2), -gz1*B3, 0, 0],
                                                 [0, 0, gz2*B3, g2*(B1+1j*B2)],
                                                 [0, 0, g2*(B1-1j*B2), -gz2*B3]])

    return zeeman

zeeman = make_Zeeman(Bx)

@njit
def cal_hamk_mini(hamk):

    hamk_mini = hamk[:num_band, :num_band]

    return hamk_mini

@njit
def make_hamk(kx, ky):
    
    hamk = cal_hamk(kx, ky)
    hamk = hamk+zeeman
    hamk_mini = cal_hamk_mini(hamk)

    return hamk_mini


if __name__ == "__main__":

    # lattice constant
    a = 12.9077420000000007
    c = 25.9834260000000015
    a1 = np.array([a,  0,  0])
    a2 = np.array([0,  a,  0])
    a3 = np.array([0,  0,  c])
    vol = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi*(np.cross(a2, a3))/vol
    b2 = 2*np.pi*(np.cross(a3, a1))/vol
    b3 = 2*np.pi*(np.cross(a1, a2))/vol
    # high symmetry point
    Gamma = np.array([0, 0])
    X = 1/2*b3
    Z = 1/2*b1+1/2*b2-1/2*b3
    # b1 b2 after rotation 
    # b1 = b1[:2]@rot_mat(theta)
    # b2 = b2[:2]@rot_mat(theta)
    b1 = b1[:2]
    b2 = b2[:2]
    print("b1 b2", b1, b2)
    X_n = -1/10*b2
    X_p = 1/10*b2
    print("Xn, Xp: ", X_n, X_p)

    n_k = 500
    high_symm_points = {'-X':X_n, 'Gamma':Gamma, 'X':X_p}
    kline, kmesh = set_tb_disp_kmesh(n_k, high_symm_points)
    eig_vals = []

    for kpnt in kmesh:
        k1, k2 = kpnt
        hamk = make_hamk(k1, k2)
        eig, _ = np.linalg.eigh(hamk)
        eig_vals.append(eig)
    eig_vals = np.array(eig_vals)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout='constrained')
    for i in range(num_band):
        ax.plot(kline, eig_vals[:, i], **{'ls':'-', 'color':'#4A90E2', 'lw':1})
    ax.set_xlim(0, kline[-1])
    ax.set_ylim([-0.025, 0.025])
    knorm = np.round(np.sqrt(X_n[0]**2+X_n[1]**2),decimals=3)
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k]])
    ax.set_xticklabels(["-"+str(knorm), "$k_y$ (ang$^{-1}$)", str(knorm)])
    ax.axvline(x=kline[n_k], color="grey", lw=0.8, ls='--')
    ax.set_ylabel("Energy (eV)")
    ax.set_title("[112] 20 nm")
    #plt.savefig("112_r2.png", dpi=500)
    #np.save("kline.npy", kline)
    #np.save("band_112_"+str(Bx)+"T.npy", eig_vals)
    plt.show()



