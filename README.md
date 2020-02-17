<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Pytorch iAlgebra

**[Document]()|[Paper]()|[References]()**

*Pytorch iAlgebra* is an interactive interpretation library for deep learning on [Pytorch](https://pytorch.org).


Pytorch iAlgebra provides an interactive frame for interpreting a group of deep leanring models using a set of interpretation methods.

----
**iAlgebra Operations**
----





**Operators**




*Identity*


$$
[\phi(x)]_{i}=\frac{1}{d} \sum_{k=0}^{d-1} \mathbb{E}_{I_{k}}\left[f\left(x_{I_{k} \cup\{i\}}\right)-f\left(x_{I_{k}}\right)\right]
$$


*Projection*

$$
\left[\Pi_{w}(x)\right]_{i}=\left\{\begin{array}{cc}{\frac{1}{|w|} \sum_{k=0}^{|w|-1} \mathbb{E}_{I_{k}}\left[f\left(x_{I_{k} \cup\{i\}}\right)-f\left(x_{I_{k}}\right)\right]} & {i \in w} \\ {0} & {i \notin w}\end{array}\right.
$$


*Selection*
$$
\left[\sigma_{l}(x)\right]_{i}=\left[\phi\left(x ; \bar{x}, f_{l}\right)\right]_{i}
$$

*Join*

$$
\left[x \bowtie x^{\prime}\right]_{i}=\frac{1}{2}\left([\phi(x ; \bar{x}, f)]_{i}+\left[\phi\left(x^{\prime} ; \bar{x}, f\right)\right]_{i}\right)
$$


*Anti-Join*

$$
\left[x \diamond x^{\prime}\right]_{i}=\left(\left[\phi\left(x ; x^{\prime}, f\right)\right]_{i},\left[\phi\left(x^{\prime} ; x, f\right)\right]_{i}\right)
$$


----
**Supportive DNN and Interpretation Models**
----

**DNN Models**

Model Performance on dataset *Mnist*

| Dataset     |                Models  |         |
| ----------- | -----------    | -----------     |
| Mnist       | LeNet-L1  | LeNet-L2   |
| Accuracy    | 98.866%        |99.020%          |


Model Performance on dataset *Cifar10*

| Dataset     |                Models  |         |
| ----------- | -----------    | -----------     |
| Cifar10     | Vgg19 -L1      | Vgg19-L2        |
| Accuracy    | 98.866%        | 99.020%          |



**Interpretation Methods**


In detail, we implement the following interpretation methods as the *identity* in Pytorch-iAlgebra.

* **GradSaliency** from Simonyan *et al.*:[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034.pdf) (CVPR 2013)

* **SmoothGrad** from Smilkov *et al.*:[SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf)

* **Mask** from Fong *et al.*:[Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/pdf/1704.03296.pdf) (ICCV 2017)

* **GradCam** from Selvaraju *et al.*: [Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) (ICCV 2017)

* **GuidedBackpropGrad** from Springenberg *et al.*:[Striving for Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf) (ICLR 2015)

<!-- * **GuidedBackpropSmoothGrad** -->

------


**Installation**
----
Library dependencies for the *Pytorch-iAlgebra*. Before installation, you need to install these with

```python
$ pip install -r requirements.txt
```

Then *Pytorch-iAlgebra* can be installed by:

```python
$ pip install pytorch-ialgebra
```






<!-- 

A sample project that exists as an aid to the [Python Packaging User
Guide][packaging guide]'s [Tutorial on Packaging and Distributing
Projects][distribution tutorial].

This project does not aim to cover best practices for Python project
development as a whole. For example, it does not provide guidance or tool
recommendations for version control, documentation, or testing.

[The source for this project is available here][src].

Most of the configuration for a Python project is done in the `setup.py` file,
an example of which is included in this project. You should edit this file
accordingly to adapt this sample project to your needs.

----

This is the README file for the project.

The file should use UTF-8 encoding and can be written using
[reStructuredText][rst] or [markdown][md use] with the appropriate [key set][md
use]. It will be used to generate the project webpage on PyPI and will be
displayed as the project homepage on common code-hosting services, and should be
written for that purpose.

Typical contents for this file would include an overview of the project, basic
usage examples, etc. Generally, including the project changelog in here is not a
good idea, although a simple “What's New” section for the most recent version
may be appropriate.

[packaging guide]: https://packaging.python.org
[distribution tutorial]: https://packaging.python.org/tutorials/packaging-projects/
[src]: https://github.com/pypa/sampleproject
[rst]: http://docutils.sourceforge.net/rst.html
[md]: https://tools.ietf.org/html/rfc7764#section-3.5 "CommonMark variant"
[md use]: https://packaging.python.org/specifications/core-metadata/#description-content-type-optional -->


