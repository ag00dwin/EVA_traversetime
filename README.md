# EVA_traversetime

## Context
Meili-I was the first analogue  mission by [Space Health Research](https://spacehealthresearch.com/). (SHR) and represents the UK’s first high-fidelity analogue astronaut crewed mission. The Geological Surveys of Impact Regions (GSIR) study of Meili-I facilitated the testing of important logistical considerations for planetary exploration completed during Extravehicular Activity (EVA). 

This repository stores Python code used to process GPS tracks collected using the smartphone application [Avenza](https://www.avenza.com/avenza-maps/).

## Referencing
If using this code, please reference: ``ag00dwin/EVA_traversetime`` 

Part of this work was presented at LPSC 2024, Houston TX: 
>Goodwin, A., Hammet, M. and Harris, M., 2024. Logistical Considerations for Astronaut Traverses and Geological Data Collection from Meili-I: The UK's First High Fidelity Planetary Exploration Analogue Astronaut Mission. LPI Contributions, 3040, p.1030.

## License
The project is licensed under the GNU-v3 license.

## Installation 
No installation is required. Scripts are short and designed to be adapted and applied to various problems. Python scripts can be downloaded from this repository and intergrated into pre-existing scripts where such functionality is required.

## Python Scripts
>NOTE, here we define “slope” as a vector quantity evaluated in the direction of travel — specifically, the tangent of the change in height experienced for any given unit of horizontal distance covered. 

**_traverse walkspeed-slope modelling.py** 
Calculates slope for GPS location fixes within a recorded traverse and plots this as historgram. Fits a generalised version of [Toblers Hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) (THF) to model how walking velocity varies with slope. 

<img src="https://github.com/ag00dwin/EVA_traversetime/blob/main/_output/_traverse%20walkspeed-slope%20modelling_plot.png" width="400">

**_traverse dijkstra paths.py** 
Runs a path-finding algorithm on the DEM raster weighted to travel time, estimated using a [Toblers Hiking function](https://en.wikipedia.org/wiki/Tobler%27s_hiking_function) (THF) model. By inputting the starting (A) and finishing (B) locations for a one-way route, the most time-optmial path will be generated as well as a heatmap (with or without contours) for time to reach all accessible grid pixels on the DEM. The code used for dijkstra’s algorithm upon an adjacency matrix (the DEM raster) is from an [article](https://judecapachietti.medium.com/dijkstras-algorithm-for-adjacency-matrix-in-python-a0966d7093e8) by Jude Capachietti. 

<img src="https://github.com/ag00dwin/EVA_traversetime/blob/main/_output/da_map_(100%2C%20371)_plot%20annotations.png" width="400">
