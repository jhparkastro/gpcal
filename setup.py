from distutils.core import setup
setup(
  name = 'gpcal',         # How you named your package folder (MyLib)
  packages = ['gpcal'],   # Chose the same as "name"
  version = '0.9.5.5',      # Start with a small number and increase it with every change you make
  license='gpl-2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A generalized instrumental polarization calibration pipeline for VLBI data',   # Give a short description about your library
  author = 'Jongho Park',                   # Type in your name
  author_email = 'jpark@asiaa.sinica.edu.tw',      # Type in your E-Mail
  url = 'https://github.com/jhparkastro/gpcal',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/jhparkastro/gpcal/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['VLBI', 'polarimetry', 'astronomy', 'calibration'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
	  'pandas',
	  'astropy',
	  'matplotlib',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',   # Again, pick a license
    'Programming Language :: Python :: 2.7',      #Specify which pyhton versions that you want to support
  ],
)
