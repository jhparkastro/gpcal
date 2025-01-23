GPCAL
===================

GPCAL is a Python module for instrumental polarization calibration in VLBI data. Its design aims to improve calibration accuracy by enabling users to simultaneously fit the model to data from multiple calibrators and incorporate the linear polarization structures of these calibrators, rather than relying on the conventional similarity assumption.

Installation
------------
GPCAL is based on AIPS and uses ParselTongue, a Python interface to AIPS (`Kettenis et al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ASPC..351..497K>`_), to run AIPS tasks. ParselTongue can be easily installed by following the instructions `here <https://www.jive.eu/jivewiki/doku.php?id=parseltongue:parseltongue>`_.

GPCAL is available on `PyPi <https://pypi.org/project/gpcal/>`_ and can be installed via pip.

.. code-block:: bash

    pip install gpcal

It should install most of the required libraries automatically (`astropy <http://www.astropy.org/>`_, `numpy <http://www.numpy.org/>`_, `pandas <http://www.pandas.pydata.org/>`_ , `matplotlib <http://www.matplotlib.org/>`_,  `scipy <http://www.scipy.org/>`_).

Note that ParselTongue is based on Python 2. One of the easiest ways to avoid any conflicts in Python versions among different softwares is to create a new Anaconda environment for Python 2 and install ParselTongue there.

.. code-block:: bash

    conda create -n myenv python=2.7
    conda activate myenv
    conda install -c kettenis parseltongue

Try to run ParselTongue in the environment and import gpcal.

Note: Python 2 is not supported on Macs with M1 or later chips running macOS versions higher than 12.3. Consequently, using GPCAL on the latest Mac systems can be challenging. Although we attempted to transition to Python 3, a critical bug in some key functions of ParselTongue3 has not been resolved by the developer. For Mac users who still wish to install GPCAL, a workaround method is available `here <https://docs.google.com/document/d/1gVV6uuZXVAMGbBygtg7JBudkeF1HOBFeUQMYipeDzrE/edit?usp=sharing>`_, which has been tested successfully on M1/OS 14.2.1 and M3/OS 14.6.1.

Tutorial
-------------
Example data and scripts are available in the "examples" folder. Tutorial slides using one of these data sets can be found `here <https://docs.google.com/presentation/d/16Rhb2WOrtrEJIjXL83XM0uWXQ3YVmkc_F8tVfLpJuK8/edit?usp=sharing>`_. Additionally, we have implemented new methods to calibrate frequency and time-dependent polarimetric leakages in VLBI data. For more details, readers are referred to `Park et al. 2023a <https://ui.adsabs.harvard.edu/abs/2023ApJ...958...27P/abstract>`_ and `Park et al. 2023b <https://ui.adsabs.harvard.edu/abs/2023ApJ...958...28P/abstract>`_. The tutorials for these methods can be found here (frequency-dependent leakage calibration method) and `here <https://docs.google.com/presentation/d/1Jc4_FRA_gLOBHRGQq6sVYI7OLu647e2sCZ9MobeEQcg/edit?usp=sharing>`_ (time-dependent leakage calibration method).

Publications that have utilized GPCAL
------------
If you use GPCAL in your publication, please cite `Park et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...906...85P/abstract>`_.

Let us know if you use GPCAL in your publication and we'll list it here!

- GPCAL: a generalized calibration pipeline for instrumental polarization in VLBI data, `Park et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...906...85P/abstract>`_ 
- Jet Collimation and Acceleration in the Giant Radio Galaxy NGC 315 `Park et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...909...76P/abstract>`_ 
- First M87 Event Horizon Telescope Results. VII. Polarization of the Ring `EHT Collaboration et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...910L..12E/abstract>`_ 
- A Revised View of the Linear Polarization in the Subparsec Core of M87 at 7 mm `Park et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021ApJ...922..180P/abstract>`_ 
- Probing the Heart of Active Narrow-line Seyfert 1 Galaxies with VERA Wideband Polarimetry `Takamura et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...952...47T/abstract>`_ 
- Calibrating VLBI Polarization Data Using GPCAL. I. Frequency-dependent Calibration `Park et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...958...27P/abstract>`_ 
- Calibrating VLBI Polarization Data Using GPCAL. II. Time-dependent Calibration `Park et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...958...28P/abstract>`_ 
- First GMVA observations with the upgraded NOEMA facility: VLBI imaging of BL Lacertae in a flaring state `Kim et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023A%26A...680L...3K/abstract>`_ 
- Up around the bend: A multiwavelength view of the quasar 3C 345 `Roder et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024A%26A...684A.211R/abstract>`_ 
- Spectral and magnetic properties of the jet base in NGC 315 `Ricci et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025A%26A...693A.172R/abstract>`_ 

License
-------
GPCAL is licensed under GPLv2+. See LICENSE.txt for more details.


