# ANM Mixture Model (ANM-MM)

Python implementation of the following paper

Hu, Shoubo, et al. "Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models." Advances in Neural Information Processing Systems. 2018.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- NumPy
- SciPy
- scikit-learn

We test the code using **Anaconda 4.3.30 64-bit for python 2.7** on Windows 10. Any later version should still work perfectly. The download page of Anaconda is [here](https://www.anaconda.com/download/).

## Running the tests

After installing all required packages, you can run *test.py* to see whether **ANM-MM** could work normally.

The test code does the following:
1. it generate 100 observations from two exponential functions.
(Data is organized in an 100-by-2 *numpy array*. The first column is the cause $X$ and the second is the effect $Y$.)
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

Input of function **ANMMM_cd()** and **ANMMM_clu()**

| Argument  | Description  |
|---|---|
|data | Numpy array with 2 columns and any number of rows. Rows represent i.i.d. samples, The first column is the hypothetical $X$ and the second is the hypothetical $Y$.|
|label | List of true labels of each observation. |
|lda |The parameter $\lambda$ which controls the importance of HSIC term. |

Output of function **ANMMM_cd()**

| Argument  | Description  |
|---|---|
|direction | 1  - the first column is the cause;<br/>-1 - the second column is the cause;<br/>0  - can not tell. |

Output of function **ANMMM_clu()**

| Argument  | Description  |
|---|---|
|labels | List of estimated clustering labels of each observation.|

## Authors

* **Shoubo Hu** - shoubo DOT sub AT gmail DOT com
* **Zhitang Chen** - chenzhitang2 AT huawei DOT com

See also the list of [contributors](https://github.com/amber0309/GPPOM/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to [James Hensman](http://jameshensman.github.io/) for his elegant [GP-LVM code](https://github.com/jameshensman/pythonGPLVM).
