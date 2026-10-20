The KSEIR model is a Degree-Based Mean-Field model used to implement quarentines to mitigate epidemic spreading.

KSEIR_functions.py defines the functions to be used throughout the project.

KSEIR_medidas.py implements the basic quarantine  measures, with fixed gamma, to measure key indicators such as the infection peak, the time to peak and epidemic size. 

KSEIR_aislamiento_bimodal.py adds more intrincated degree distributions to account for more complex behaviours, such as essential workers that do not isolate during quarantine. 

KSEIR_aislamiento_ciclico.py changes the strategy of quarantine to make it cyclical with a pre-defined duration and resting periods. 
