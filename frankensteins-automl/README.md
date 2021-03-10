# frankensteins-automl
This is the reference implementation of my Master Thesis [Optimizer Ensembles for Automated Machine Learning](https://github.com/Berberer/master-thesis).

**Simple usage**:
```python
from frankensteins_automl.frankensteins_automl import (
    FrankensteinsAutoMLConfig,
    FrankensteinsAutoML,
)

config = FrankensteinsAutoMLConfig()
# Either use an ARFF file as input
config.data_input_from_arff_file("<path/to/data/arff/file>", <target_column_index>)
# Or alternatively, provide the data as two arrays yourself
config.direct_data_input(<data_array>, <target_class_array>)

automl = FrankensteinsAutoML(config)
results = automl.run()
pipeline = results["pipeline_object"]
score = results["search_score"]
```
For other configuration possibilities please refer to the  `FrankensteinsAutoMLConfig` class.
The default timeouts there are really short, so you probably want to adjust them for any non-toy dataset.

**Visualize search**:
1. Add the following line to the config:
```python
config.event_send_url = "http://localhost:3000/event"
```
2. Run `yarn install` inside of the `search_visualization/` folder
3. Run `yarn electron:serve` inside of the `search_visualization/` folder **before** you start frankensteins-automl
