#       Layer: 1     2      3      4      5      6     #
#------------------------------------------------------#
skip_ss =     [0,    0,     0,     0,     0,     0]    # Skips image sampling
skip_s =      [0,    0,     0,     0,     0,     0]    # Skip S layers
skip_c =      [1,    0,     0,     0,     0,     0]    # Skip C layers
skip_t =      [0,    0,     0,     0,     0,     0]    # Skip Training
skip_i =      [0,    0,     0,     0,     0,     0]    # Skip Inference
skip_p =      [0,    0,     0,     0,     0,     0]    # Skip Pooling

num_bases =   [1e2,  2e2,   4e2,   5e2,   5e2,   5e2]  # Number of bases
bases_size =  [10,   10,    10,    10,    10,    10]   # Size of bases

num_samp =    [1e4,  2e4,   4e4,   5e4,   5e4,   5e4]  # Number of samples

s_stride =    [2,    2,     1,     1,     1,     1]    # Stride for S layers
c_stride =    [1,    1,     1,     1,     1,     1]    # Stride for C layers

pool_ratio =  [2,    2,     2,     2,     2,     2]   # Pooling Ratio