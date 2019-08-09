# ANM Mixture Model (ANM-MM)

Python implementation of the following paper

[Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models.](http://papers.nips.cc/paper/7767-causal-inference-and-mechanism-clustering-of-a-mixture-of-additive-noise-models)  
Hu, Shoubo, Zhitang Chen, Vahid Partovi Nia, Laiwan Chan, and Yanhui Geng.  
*Advances in Neural Information Processing Systems.* (**NeurIPS**) 2018.

## Prerequisites
- numpy
- scipy
- sklearn

We test the code using **Anaconda 4.3.30 64-bit for python 2.7** on Windows 10. Any later version should still work perfectly.

## Running the tests

After installing all required packages, you can run *test.py* to see whether **ANM-MM()** could work normally.

The test code does the following:
1. it generates 100 observations (a (100, 2) *numpy array*) from two exponential functions. The first column is the cause `X` and the second is the effect `Y`.
2. ANM-MM is applied on the generated data to first conduct clustering and then infer the causal direction.

## Apply **ANM-MM** on your data

### Usage

Import **ANM-MM** using

```python
from ANMMM import ANMMM_cd, ANMMM_clu
```

Apply **ANM-MM** on your data

```python
# causal inference
direction = ANMMM_cd(data, lda)

# mechanism clustering
labels = ANMMM_clu(data, label, lda)
```

### Description

Input of function `ANMMM_cd()` and `ANMMM_clu()`

| Argument  | Description  |
|---|---|
|data | Numpy array with 2 columns and any number of rows. Rows represent i.i.d. samples, The variables in the first and second column are called `X` and `Y`, respectively.|
|label | List of true labels of each observation. |
|lda |The parameter `λ` which controls the importance of HSIC term. |

Output of function `ANMMM_cd()`

| Argument  | Description  |
|---|---|
|direction | 1  - the first column is the cause;<br/>-1 - the second column is the cause;<br/>0  - can not tell. |

Output of function `ANMMM_clu()`

| Argument  | Description  |
|---|---|
|labels | List of estimated clustering labels of each observation.|

## Authors

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com
* **Zhitang Chen** - chenzhitang2 [at] huawei [dot] com

See also the list of [contributors](https://github.com/amber0309/GPPOM/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to [James Hensman](http://jameshensman.github.io/) for his elegant [GP-LVM code](https://github.com/jameshensman/pythonGPLVM).
