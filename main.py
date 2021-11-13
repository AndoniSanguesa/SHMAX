from parameters import *
from SHMAX_C import SHMAX_C
from SHMAX_2DS import SHMAX_2Ds
from SHMAX_3DS import SHMAX_3Ds
import time

# Layer 1
print("***  S0 LAYER  ***")
if not skip_s[0]:
    SHMAX_2Ds(skip_ss[0], skip_t[0], skip_i[0], num_bases[0], bases_size[0], num_samp[0], s_stride[0], "pydata/TRAIN/", "S1Result/")

print("***  C0 LAYER  ***")
if not skip_c[0]:
    SHMAX_C(skip_p[0], pool_ratio[0], c_stride[0], "S1Result/", "C1Result/")

# Layers 2-6
for i in range(1,6):
    print(f"*** S{i} LAYER ***")
    if not skip_s[i]:
        SHMAX_3Ds(skip_ss[i], skip_t[i], skip_i[i], num_bases[i], bases_size[i], num_samp[i], s_stride[i], f"C{i}Result/", f"S{i+1}Result/")

    print(f"*** C{i} LAYER ***")
    if not skip_c[i]:
        SHMAX_C(skip_p[i], pool_ratio[i], c_stride[i], f"S{i+1}Result/", f"C{i+1}Result/")