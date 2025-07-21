{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u23433 \u35013  mpmath\u65288 Colab \u38656 \u35201 \u65292 \u26412 \u22320 \u24050 \u23433 \u35013 \u21487 \u36339 \u36807 \u65289 \
!pip install mpmath --quiet\
\
# \uc0\u23548 \u20837 \u24211 \
import numpy as np\
import matplotlib.pyplot as plt\
from mpmath import mp, mpf, cos, log, pi, zetazero\
from scipy.stats import pearsonr  # \uc0\u29992 \u20110 \u30456 \u20851 \u20998 \u26512 \u65288 \u21487 \u36873 \u65289 \
\
# \uc0\u35774 \u32622 \u39640 \u31934 \u24230 \
mp.dps = 50\
\
# --- \uc0\u33719 \u21462 \u21069  N \u20010  Riemann \u38646 \u28857 \u30340 \u34394 \u37096  \u947 \u8345  ---\
def get_gamma_n(N):\
    return [mp.im(zetazero(n)) for n in range(1, N + 1)]\
\
# --- \uc0\u26500 \u36896  \u966 (n) ---\
def phi(n, gamma_n):\
    return (4 / (pi * n)) * (cos(log(gamma_n[n - 1])) + 1.1)\
\
# --- \uc0\u26500 \u36896 \u32467 \u26500 \u29109  H(n) ---\
def H(phi_n):\
    return np.log(1 + phi_n**2)\
\
# --- \uc0\u26500 \u36896 \u32467 \u26500 \u27604 \u29575 \u20989 \u25968  K(n) ---\
def compute_K(phi_vals, H_vals):\
    log_phi = np.log(np.abs(phi_vals))\
    log_H = np.log(H_vals)\
    \
    # \uc0\u19968 \u38454 \u23548 \u25968 \u65288 \u26377 \u38480 \u24046 \u20998 \u65289 \
    d_log_phi = np.gradient(log_phi)\
    d_log_H = np.gradient(log_H)\
    \
    return d_log_phi / d_log_H\
\
# --- \uc0\u20027 \u27969 \u31243  ---\
N = 200  # \uc0\u21487 \u35843 \u25972 \u20026 300\u25110 \u26356 \u22823 \u65288 \u27880 \u24847 \u35745 \u31639 \u26102 \u38388 \u65289 \
gamma_list = get_gamma_n(N)\
gamma_array = np.array([float(g) for g in gamma_list])  # \uc0\u36716 \u25442 \u20026 float\u25968 \u32452 \u65292 \u20197 \u25903 \u25345 NumPy\u20989 \u25968 \u22914 arctan\
phi_vals = np.array([float(phi(n + 1, gamma_list)) for n in range(N)])\
H_vals = H(phi_vals)\
K_vals = compute_K(phi_vals, H_vals)\
\
# --- \uc0\u35745 \u31639  \u966  / \u8730 H \u30340 \u20998 \u24067  (R(n)) ---\
ratio_vals = phi_vals / np.sqrt(H_vals)  # \uc0\u20551 \u35774 phi>0\u65292 \u26080 \u38656 abs\
\
# --- \uc0\u35745 \u31639 \u20960 \u20309 \u30456 \u20301  (\u35770 \u25991 Section 4) ---\
theta_n = np.arctan(2 * gamma_array)\
tau_n = np.pi / 2 - theta_n\
\
# --- \uc0\u36755 \u20986 \u32479 \u35745 \u20449 \u24687 \u65306 K(n) ---\
print(f"\uc0\u9989  \u24179 \u22343  K(n) \u8776  \{np.mean(K_vals):.5f\}")\
print(f"\uc0\u963 (K) \u8776  \{np.std(K_vals):.5f\}")\
print(f"min(K) \uc0\u8776  \{np.min(K_vals):.5f\}, max(K) \u8776  \{np.max(K_vals):.5f\}")\
\
# --- \uc0\u36755 \u20986 \u32479 \u35745 \u20449 \u24687 \u65306 R(n) ---\
print(f"\uc0\u9989  \u24179 \u22343  R(n) \u8776  \{np.mean(ratio_vals):.5f\}")\
print(f"\uc0\u963 (R) \u8776  \{np.std(ratio_vals):.5f\}")\
print(f"min(R) \uc0\u8776  \{np.min(ratio_vals):.5f\}, max(R) \u8776  \{np.max(ratio_vals):.5f\}")\
\
# --- \uc0\u36755 \u20986 \u20960 \u20309 \u30456 \u20301 \u32479 \u35745 \u65288 \u31034 \u20363 \u65289  ---\
print(f"\uc0\u9989  \u24179 \u22343  \u964 _n \u8776  \{np.mean(tau_n):.5f\}")\
print(f"\uc0\u963 (\u964 _n) \u8776  \{np.std(tau_n):.5f\}")\
print(f"min(\uc0\u964 _n) \u8776  \{np.min(tau_n):.5f\}, max(\u964 _n) \u8776  \{np.max(tau_n):.5f\}")\
\
# --- \uc0\u21487 \u36873 \u65306 \u964 _n \u19982  \u966 (n) \u30340 Pearson\u30456 \u20851 \u31995 \u25968  ---\
corr, _ = pearsonr(tau_n, phi_vals)\
print(f"\uc0\u964 _n \u19982  \u966 (n) \u30340 Pearson\u30456 \u20851 \u31995 \u25968 : \{corr:.5f\}")\
\
# --- \uc0\u21487 \u35270 \u21270  K(n) ---\
plt.figure(figsize=(7, 5))\
plt.plot(range(1, N + 1), K_vals, label=r'$K(n) = \\frac\{d \\log|\\phi|\}\{d \\log H\}$')\
plt.axhline(0.5, color='green', linestyle='--', label='K = 0.5')\
plt.axhline(1, color='red', linestyle='--', label='K = 1')\
plt.title('Entropy / Structure Ratio K(n)')\
plt.xlabel('n')\
plt.ylabel('K(n)')\
plt.legend()\
plt.grid(True)\
plt.show()\
\
# --- \uc0\u21487 \u35270 \u21270  R(n) ---\
plt.figure(figsize=(7, 5))\
plt.plot(range(1, N + 1), ratio_vals, label=r'$R(n) = \\phi / \\sqrt\{H\}$')\
plt.axhline(1, color='red', linestyle='--', label='R = 1')\
plt.title('Distribution of R(n)')\
plt.xlabel('n')\
plt.ylabel('R(n)')\
plt.legend()\
plt.grid(True)\
plt.show()\
\
# --- \uc0\u21487 \u35270 \u21270  \u964 _n ---\
plt.figure(figsize=(7, 5))\
plt.plot(range(1, N + 1), tau_n, label=r'$\\tau_n = \\pi/2 - \\arctan(2 \\gamma_n)$')\
plt.title('Structural Residue \\tau_n')\
plt.xlabel('n')\
plt.ylabel('\\tau_n')\
plt.legend()\
plt.grid(True)\
plt.show()\
}