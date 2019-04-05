# 3DCORE

Basic implementation of the prototype described in [Möstl et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017SW001735).

## Installing

Install by cloning the repository:

```
https://github.com/IWF-helio/3DCORE.git
```

The code is implemented as a single package called `mfr3dcore`. It is highly recommended to use a python virtual 
environment. Make sure you have the latest version of `virtualenv` installed and create a new virtual environment.

```
virtualenv -p /usr/bin/python3.7 venv
```

Enter the virtual environment and install all required packages:


```
source venv/bin/activate
pip install -r 3DCORE/requirements.txt
```

The `mfr3dcore` package can be installed using:

```
pip install -e 3DCORE
```

You can now use the package by using `import mf3dcore`. 

If you want to use any SPICE kernels you will need to download the respective kernel files. If you want to download all 
kernels simply use the script `setup_spice.sh` (you need to be inside the 3DCORE folder).

For basic usage see the given examples.