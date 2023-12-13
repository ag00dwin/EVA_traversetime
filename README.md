# EVA_traversetime

## Context
Meili-I was the first analogue  mission by [Space Health Research](https://spacehealthresearch.com/). (SHR) and represents the UK’s first high-fidelity analogue astronaut crewed mission. The Geological Surveys of Impact Regions (GSIR) study of Meili-I facilitated the testing of important logistical considerations for planetary exploration completed during Extravehicular Activity (EVA). 

This repository stores Python code used to process GPS tracks collected using the smartphone application [Avenza](https://www.avenza.com/avenza-maps/).

## Referencing
If using this code, please reference: ``ag00dwin/EVA_traversetime`` — details regarding conference abstracts to be added soon

## License
The project is licensed under the GNU-v3 license.

## Installation 
No installation is required. Scripts are short and designed to be adapted and applied to various problems. Python scripts can be downloaded from this repository and intergrated into pre-existing scripts where such functionality is required.

## Python Scripts

**_activity stop modelling.py** 
Uses a traverse speed threshold and cluster anlalysis in XYtime to locate within the GPS tracks when breaks were taken. Calculates a historgram of the length of time of each break. 

<img src="https://github.com/ag00dwin/EVA_traversetime/blob/main/_output/_activity%20stop%20modelling_hist.png" width="400">

**_traverse walkspeed-slope modelling.py** 
Calculates relative slope and absolate slope for intervals of the GPS tracks and plots this as historgram. Fits functions (including [Toblers Hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function)) to relative and absolate slope to model how walking speed varies with terrain. 

<img src="https://github.com/ag00dwin/EVA_traversetime/blob/main/_output/_traverse%20walkspeed-slope%20modelling_plot.png" width="400">

**_traverse dijkstra paths.py** 
Runs a path-finding algorithm from the relative slope exponential walking model (from ``_traverse walkspeed-slope modelling``) based upon [Toblers Hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function), and uses this to calculate the most efficent paths from A to B on the island DEM. Outputs the path itself as a .csv file, as well as a heatmap (with or without contours) for time to reach all accessible grid pixels on the DEM. 

<img src="https://github.com/ag00dwin/EVA_traversetime/blob/main/_output/da_map_(100%2C%20371)_plot%20annotations.png" width="400">
