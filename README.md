<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script> -->

# Pytorch-iAlgebra

**[Document]()|[Paper]()|[References]()**

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
***Pytorch-iAlgebra* Introduction**
----

*Pytorch-iAlgebra* can be used in two way:

>**Web-based Example**

Run the server by 
```console
$ cd pytorch-ialgebra/frontend_demo
$ python server.py -s "http://your-server-address.edu" -p port -f "./ialgebra.html"
```

Then you can open the User Interface at: `http://your-server-address.edu:port/./ialgebra.html`. 

![Demo of iAlgebra](https://github.com/huashen218/pytorch-ialgebra/blob/master/frontend_demo/ialgebra_ui_demo.png?raw=true "Demo of iAlgebra")




>**Programming-based Example**

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

<!-- * **GuidedBackpropSmoothGrad** -->







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


