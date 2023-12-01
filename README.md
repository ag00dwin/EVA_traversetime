# EVA_traversetime

_activity stop modelling : 
Uses a speed threshold and cluster anlalysis in X/Y/time to locate within the GPS tracks when breaks were taken. Calculates a historgram of the length of time of each break. 

_traverse walkspeed-slope modelling: 
Calculates relative slope and absolate slope for intervals of the GPS tracks and plots as historgram. Fits functions to relative and absolate slope to model walking speed. 

_traverse walkspeed_slope model-comparison:
Applies the walking speed models (from _traverse walkspeed-slope modelling) for relative and absolute slopes and compares to actual GPS data to asses (visual comparison) the goodness of the fit. 

_traverse dijkstra paths:
Runs a path finding algorithm from relative slope walking model (from _traverse walkspeed-slope modelling) and uses this to calculate the most efficent paths from A to B on the island DEM. Outputs the path itself as well as a heatmap for time to reach all accessible grid pixels on the DEM.