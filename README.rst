GPCAL
===================

GPCAL is a Python module for instrumental polarization calibration in VLBI data. It was designed to enhance the calibration accuracy by enabling users to fit the model to multiple calibrators data simultaneously and to take into account the calibrators linear polarization structures instead of using the conventional similarity assumption. 

Installation
------------
GPCAL is based on AIPS and uses ParselTongue, a Python interface to AIPS (`Kettenis et al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ASPC..351..497K>`_), to run AIPS tasks. ParselTongue can be easily installed by following the instructions `here <http://old.jive.nl/jivewiki/doku.php?id=parseltongue:parseltongue>`_.

GPCAL is available on `PyPi <https://pypi.org/project/gpcal/0.1.1.26/>`_ and can be installed via pip.

.. code-block:: bash

    pip install gpcal

It should install most of the required libraries automatically (`astropy <http://www.astropy.org/>`_, `numpy <http://www.numpy.org/>`_, `pandas <http://www.pandas.pydata.org/>`_ , `matplotlib <http://www.matplotlib.org/>`_,  `scipy <http://www.scipy.org/>`_).

Note that ParselTongue is based on Python 2. One of the easiest ways to avoid any conflicts in Python versions among different softwares is to create a new Anaconda environment for Python 2 and install ParselTongue there.

.. code-block:: bash

    conda create -n myenv python=2.7
    conda activate myenv
    conda install -c kettenis parseltongue

Try to run ParselTongue in the environment and import gpcal. If users encounter an error message of "...no module named _bsddb." in astroplan, then install bsddb via Anaconda in the environment.

.. code-block:: bash

    conda install -c free bsddb


Tutorial
-------------
There are example data and scripts in the "examples" folder. Tutorial slides using one of those data can be found `here <https://docs.google.com/presentation/d/1RRiT4r6H7yeu8ErvdhLN_f8XoMGg0QU3kO_WVWr85W8/edit?usp=sharing>`_.

Some publications that use GPCAL
------------
If you use GPCAL in your publication, please cite `Park et al. 2020 <>`_.

Let us know if you use GPCAL in your publication and we'll list it here!

- GPCAL: a generalized calibration pipeline for instrumental polarization in VLBI data, `Park et al. 2020 <>`_ 


License
-------
GPCAL is licensed under GPLv2+. See LICENSE.txt for more details.


