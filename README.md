libffm
======

A Python wrapper for the [libffm](http://www.csie.ntu.edu.tw/~r01922136/libffm/) library.

Quick start
-----------

```
git clone git@github.com:dato-code/GraphLab-Create-SDK.git sdk
git clone git@github.com:dato-code/python-libffm.git ffm
cd ffm
make
```

To run the following examples you will also need to [register for GraphLab Create](https://dato.com/products/create/quick-start-guide.html). This software is free for non-commercial use and has a 30 day free trial otherwise.

After that, try running the basic example:
```
ipython examples/basic.py
```

If you want to try a less synthetic example, download the [1TB Criteo dataset](http://labs.criteo.com/downloads/download-terabyte-click-logs/). 
First test things out with a small sample of the dataset. 
```
gzip -cd day_0.gz| head -n 1000000 > criteo-sample.tsv
```

Next we have a sample script for performing some of the same types of feature engineering that the contest winners have been using:
```
ipython examples/criteo_process.py
```

Train a FFM model on this data.
```
ipython examples/criteo_sample.py
```

You should see something like the following (which appears to be overfitting in this example):
```
PROGRESS: iter   tr_logloss   va_logloss
PROGRESS:    0      0.12794      0.12353
PROGRESS:    1      0.10907      0.12636
PROGRESS:    2      0.09263      0.13318
PROGRESS:    3      0.07679      0.14200
PROGRESS:    4      0.06411      0.15130
PROGRESS:    5      0.05484      0.16034
...
```

Usage
-----

The package makes it easy to train models directly from [SFrames](https://dato.com/products/create/docs/generated/graphlab.SFrame.html#graphlab.SFrame). 

```
import ffm

train = gl.SFrame('examples/small.tr.sframe')
test = gl.SFrame('examples/small.te.sframe')

m = ffm.FFM(lam=.1)
m.fit(train, target='y', nr_iters=50)
yhat = m.predict(test)
```

Each column is interpreted as a separate "field" in the model. Only dict columns are currently supported, where the keys of each dict are integers that represent the feature id.

Code
----

- `libfmm.cpp`: uses C++ macros provided by [Dato's SDK](https://github.com/dato-code/GraphLab-Create-SDK) to wrap `libffm`'s methods as Python classes and methods.
- `fmm.py`: a scikit-learn-style wrapper.
- `lib/`: the [original library](http://www.csie.ntu.edu.tw/~r01922136/libffm/), where cout statements have been replaced with Dato's `progress_stream` to allow progress printing to Python.
- `examples/`: example scripts for training  models using the sample data provided with the original package as well as with data similar to Kaggle's [criteo competition](https://www.kaggle.com/c/criteo-display-ad-challenge).

More details
------------

For more on how and why we made this, see the [blog post]().

License
-------
This package provided under the 3-clause BSD [license](LICENSE).
