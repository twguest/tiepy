=====
Usage
=====

Start by importing tiepy.

.. code-block:: python

    import tiepy-xfel
    
===
API
===


Core Functions
==============

.. autofunction:: tiepy.speckle.track.fit_gaussian_and_find_peak
.. autofunction:: tiepy.speckle.track.process_subset_images
.. autofunction:: tiepy.speckle.track.process_single_image


Tracking Methods
----------------

.. autofunction:: tiepy.speckle.track.match_template
.. autofunction:: tiepy.speckle.track.calculate_correlation
.. autofunction:: tiepy.speckle.track.calculate_normalized_correlation


Phase Retrieval
===============
.. autofunction:: tiepy.speckle.phase_retrieval.kottler
.. autofunction:: tiepy.speckle.phase_retrieval.paganin_algorithm


Utilities
=========

IO
--
.. autofunction:: tiepy.speckle.io.print_h5_keys
.. autofunction:: tiepy.speckle.io.load_tiff_as_npy
.. autofunction:: tiepy.speckle.io.get_keys
.. autofunction:: tiepy.speckle.io.create_virtual_h5
.. autofunction:: tiepy.speckle.io.save_dict_to_h5
.. autofunction:: tiepy.speckle.io.load_dict_from_h5
.. autofunction:: tiepy.speckle.io.load_key_from_virtual_h5


Test Samples
------------

.. autofunction:: tiepy.speckle.objects.generate_speckle_pattern
.. autofunction:: tiepy.speckle.objects.generate_gaussian_2d
