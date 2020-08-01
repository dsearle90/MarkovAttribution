MarkovAttribution
=================


Markov Attribution is a lightweight package that allows for quick Markov Model Attribution

  - Easy to use
  - Specify custom processing of leftover removal distribution
  - Two methods of removal effect calculation available
    -- Channel synthesis
    -- Direct approximation

Requirements
-------------

Markov Attribution uses a number of open source projects to work properly

- Python3_

- Numpy_

- Pandas_

.. _Python3: https://www.python.org/download/releases/3.0/
.. _Numpy: https://pypi.org/project/numpy/
.. _Pandas: https://pypi.org/project/pandas/

Installation
---------------

.. code:: sh

   pip install MarkovAttribution




Usage
-----------------

Download the sample csv here_ .

.. _here: https://github.com/dsearle90/MarkovAttribution/blob/master/sample/paths.csv

Get started with the following code.

.. code:: python

   import pandas as pd

   import MarkovAttribution

   df = pd.read_csv('*PATH TO SAMPLE*')

   attribution = MarkovAttribution()

   ma = attribution.fit(df)

   for key, value in ma['Markov Values'].items():

       print (key.ljust(15), round(value,2))

Documentation
-----------------
Markov attribution assumes you have a pre-stitched Pandas DataFrame containing user-touchpoints in each column. Each row represents a user journey.

You can specify which columns to include as touchpoints by marking them with the (default) column prefix *T_*. (Refer to sample input.)

The following optional parameters are provided when intializing the Markov Attribution Model:

**Path Prefix**

Specify the prefix in your "touchpoint" columns

.. code:: sh

    Parameter: path_prefix (optional, default "T_")

**Conversion Column**

Specify the name of your "conversion" column. **This column should be of type string and only contain either 'conv' or 'null'**

.. code:: sh

    Parameter: conversion_col (optional, default "conv_flag")

**Removal effect calculation mode**

Options are 'approximate' and'synthesize'
Approximate uses linear algebra to directly calculate removal effects, whereas synthesize will generate synthetic user journeys based off the removal transition matrix.
'Approximate' generates much faster results*

.. code:: sh

    Parameter: removal_calc_mode (optional, default "approximate")

**Removal leftovers redistribution mode**

Options are 'null' and 'even'
When removing a channel, we must decide how to re-allocate the missing % of journeys. Null will directly re-assign any leftover probability to a non-conversion (null). Even will scale up and re-allocate across existing channels based on their current probability.

.. code:: sh

    Parameter: removal_leftover_redist (optional, default "null")

**Number of paths to synthesize**

Only required if removal_calc_mode is set to "synthesize". Smaller values speed computation but decrease accuracy.

.. code:: sh

    Parameter: synthesize_n (optional, default 20000)


Learn More
-----------------

If you'd like to learn more about Markov Modelling for media attribution, Medium_ and AnalyticsVidya_ have some great writeups.

.. _Medium: https://medium.com/@mortenhegewald/marketing-channel-attribution-using-markov-chains-101-in-python-78fb181ebf1e

.. _AnalyticsVidya: https://www.analyticsvidhya.com/blog/2018/01/channel-attribution-modeling-using-markov-chains-in-r/
