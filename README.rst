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

Another way to install GPCAL is by using `Anaconda <https://www.anaconda.com/>`_. Since ParselTongue is based on Python 2 and can be installed via Anaconda, one of the easiest ways to avoid any conflicts in Python versions among different softwares is to create a new Anaconda environment for Python 2 and install ParselTongue and GPCAL there.

.. code-block:: bash

    conda create -n myenv python=2.7
    conda activate myenv
    conda install -c kettenis parseltongue
    conda install gpcal

Try to run ParselTongue in the environment and import gpcal. If users encounter an error message of "...no module named _bsddb." in astroplan, then install bsddb via Anaconda in the environment.

.. code-block:: bash

    conda install -c free bsddb


Documentation
-------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_ .

Here are some ways to learn to use the code:

- Start with the script examples/example.py, which contains a series of sample commands to load an image and array, generate data, and produce an image with various imaging algorithms.

- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image. Note that this presentation used a previous version of the code -- some function names and prefixes may need to be updated.

Some publications that use ehtim
------------
If you use ehtim in your publication, please cite both  `Chael et al. 2016 <http://adsabs.harvard.edu/abs/2016ApJ...829...11C>`_  and  `Chael et al. 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_

Let us know if you use ehtim in your publication and we'll list it here!

- High-Resolution Linear Polarimetric Imaging for the Event Horizon Telescope, `Chael et al. 2016 <https://arxiv.org/abs/1605.06156>`_ 

- Computational  Imaging for VLBI Image Reconstruction, `Bouman et al. 2016 <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_ 

- Stochastic Optics: A Scattering Mitigation  Framework for Radio Interferometric Imaging, `Johnson 2016 <https://arxiv.org/abs/1610.05326>`_ 

- Quantifying Intrinsic Variability of  Sgr A* using Closure Phase Measurements of the Event Horizon Telescope, `Roelofs et al. 2017 <https://arxiv.org/abs/1708.01056>`_ 

- Reconstructing Video from Interferometric Measurements of Time-Varying Sources, `Bouman et al. 2017 <https://arxiv.org/abs/1711.01357>`_  

- Dynamical Imaging with Interferometry, `Johnson et al. 2017 <https://arxiv.org/abs/1711.01286>`_  

- Interferometric Imaging Directly with Closure Phases and Closure Amplitudes, `Chael et al. 2018 <https://arxiv.org/abs/1803.07088>`_

- A Model for Anisotropic Interstellar Scattering and its Application to Sgr A*, `Psaltis et al. 2018 <https://arxiv.org/abs/1805.01242>`_

- The Currrent Ability to Test Theories of Gravity with Black Hole Shadows, `Mizuno et al. 2018 <https://arxiv.org/abs/1804.05812>`_

- The Scattering and Intrinsic Structure of Sagittarius A* at Radio Wavelengths, `Johnson et al. 2018 <https://arxiv.org/abs/18008.08966>`_

- How to tell an accreting boson star from a black hole, `Olivares et al. 2018 <https://arxiv.org/abs/1809.08682>`_

- Testing General Relativity with the Black Hole Shadow Size and Asymmetry of Sagittarius A*: Limitations from Interstellar Scattering, `Zhu et al. 2018 <https://arxiv.org/abs/1811.02079>`_

- The Size, Shape, and Scattering of Sagittarius A* at 86 GHz: First VLBI with ALMA, `Issaoun et al. 2019 <https://arxiv.org/abs/1901.06226>`_


Acknowledgements
----------------
The oifits_new code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being developed and may not work with all versions of python or astropy.

The documentation is styled after `dfm's projects <https://github.com/dfm>`_ 

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.



