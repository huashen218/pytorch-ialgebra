<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script> -->

# Pytorch-iAlgebra

<!-- **[Document]()|[Paper]()|[References]()** -->

*Pytorch-iAlgebra* is an interactive interpretation library for deep learning on [Pytorch](https://pytorch.org).


*Pytorch-iAlgebra* implements i-Algebra with a User Interface (UI) for user interaction and a Pytorch Interpretable Deep Learning System (IDLS), which integrates a set of interpertation models, for processing interactive interpretation queries. Details can be found in paper:

> * i-Algebra: Towards Interactive Interpretability of Neural Nets


------
**Requirements**
----
Before installing *Pytorch-iAlgebra*, the following libraries are required:

* Python3
* Pytorch
* etc.

The `requirements.txt` file lists all python libraries that *Pytorch-iAlgebra* depends on, you can install them by using:

```console
$ pip install -r requirements.txt
```

------
***Pytorch-iAlgebra* Installation**
----

The *Pytorch-iAlgebra* can be installed:

using `pip`:
```console
$ pip install pytorch-ialgebra
```
using `source`:
```console
$ python setup.py install
```



----
***Pytorch-iAlgebra* Examples**
----

*Pytorch-iAlgebra* can be used in two ways:

>**Usage1: Web-based Example**

Run the server by 
```console
$ cd pytorch-ialgebra/frontend_demo
$ python server.py -s "http://your-server-address" -p your-port -f "./ialgebra.html"
  (e.g., python server.py -s "http://university.edu" -p 8890 -f "./ialgebra.html")
```

Then you can open the User Interface (UI) at: `http://your-server-address:port/./ialgebra.html` (The UI demo is shown below), and make interactive interpretaiton.

![Demo of iAlgebra](https://github.com/huashen218/pytorch-ialgebra/blob/master/frontend_demo/ialgebra_ui_demo.png?raw=true "Demo of iAlgebra")


>**Usage2: Programming-based Example**

```python
from ialgebra.operators import *
from ialgebra.utils.utils_model import load_model
from ialgebra.utils.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map

# define operator
_operator_class = Operator(identity_name, dataset, device = device)
operator = getattr(_operator_class, operator_name)

# interpretation
heatmap, heatmapimg = operator(bx, by, model_list)

# visualize and save image
vis_saliancy_map(heatmap, heatmapimg)
save_attribution_map(heatmap, heatmapimg, './')
```

----
**Citing this Work**
----

When using *Pytorch-iAlgebra*, please use the following citation:
```
@misc{ialgebra2020,
    title={i-Algebra: Towards Interactive Interpretability of Neural Nets},
    year={2020}
}
```