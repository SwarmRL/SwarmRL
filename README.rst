SwarmRL
-------
Python package to study particle motion with reinforcement learning.

Installation
============

.. code-block:: bash

   cd SwarmRL
   pip install -e .

This will add reference to the current directory. Therefore, when you change the
package you need not reinstall.

Getting started
===============
Go to examples and run it with espresso.

TODO
====
We need to connect observables with models. If we have a gradient model that requires
historical distances, this should be returned by the task as it is 100 % task
dependent. It also reduces the amount of stuff the user has to instantiate which is a
win-win.
