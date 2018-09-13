from __future__ import division
from ANMMM import ANMMM_cd, ANMMM_clu
from gen_syn import gen_D

# ----- generate synthetic data
# 50 points from f = np.exp( -\theta_{1} * x )
# 50 points from f = np.exp( -\theta_{2} * x )
data, label_true = gen_D([[1, 1, 2, 0, 50], [1, 2, 2, 0, 50]])

# ----- clustering
ANMMM_clu(data, label_true, 1.0)

# ----- causal inference
ANMMM_cd(data, 1.0)