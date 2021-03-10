Basics
------
Monte-Carlo tree search aided with the technique of simulated annealing to approximate global minimum where the hierarchical domain structure of the state space is paired up with an increasing importance of exploiting known routes as the algorithm proceeds.\
It is helpful when the runtime is a sensitive parameter and there is no capacity for an exhaustive search.\
Goal is to find or approximate the best combinations of certain items with respect to a specific fitness.\
The algorithm tries to walk through the most promising states under the time allocated for it.\
It gradually builts up the tree and thus the more complecated combinations by inferring the descendants' fitness based on that of their own and their ancestors, then updating every fitness on the route taken by backpropagation.

Note that these codes have been extracted from a large project, where they were tailored to perform a specific task.\
Further generalization would certainly be necessary.

Further information
-------------------
Further readings can be found in the <a href="/docs">docs</a>.

Minimum working example
-----------------------
```bash
pip install -r requirements.txt
python src/mwe.py
```

Visualization
-------------
The algorithm generates interactive graph using Bokeh.

<img src="images/bokeh.png" width=500 height=550 >
