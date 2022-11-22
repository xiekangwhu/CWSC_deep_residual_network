# CWSC_deep_residual_network

This code contains python code used to deep residual network for catchment water storage capacity (CWSC) reconstruction. Deep Residual Network (ResNet), one of the specific types of CNN method, can automatically learn features from large-scale data and generalize the results to unknown data of the same type. 

The inputs of the model include 15 inputs such as global meteorological data, soil and vegetation data, topographical data, and streamflow data. The CWSC parameters on the global grids obtained by the calibration algorithm are taken as the target labels of the model. On grids with KGE greater than 0, CWSC parameters can be obtained by calibration of the hydrological model.

Areas with KGE less than 0 are masked. On grids with KGE greater than zero, the samples are divided into training set and test set according to the ratio of 7:3. The model is run on a GPU (Nvidia Tesla V100 16GB) cluster, and takes 758 microseconds per step, for a total of about one hour. 

Major code contributor: Kang Xie (PhD Student, Wuhan University) and Shuanghong Shen (PhD Student, University of Science and Technology of China)


Abstract

Catchment water storage capacity (CWSC) links the atmosphere and terrestrial ecosystems, which is required as spatial parameters for geoscientific models. However, there are currently no available common datasets of the CWSC on a global scale, especially for hydrological models since conventional evapotranspiration-derived estimates cannot represent the extra storage capacity for the lateral flow and runoff generation. Here, we produce a dataset of the CWSC parameter for global hydrological models. Joint parameter calibration of three commonly used monthly water balance models provides the labels for a deep residual network. The global CWSC is constructed based on the deep residual network at 0.5° resolution by integrating 15 types of meteorological forcings, underlying surface properties, and runoff data. CWSC products are validated with the spatial distribution against root zone depth datasets and validated in the simulation efficiency on global grids and typical catchments from different climatic regions. We provide the global CWSC parameter dataset as a benchmark for geoscientific modelling by users.


Citations

If you find our code to be useful, please cite the following papers:

Xie, K. et al. Identification of spatially distributed parameters of hydrological models using the dimension-adaptive key grid calibration strategy - ScienceDirect. Journal of Hydrology 598, doi:10.1016/j.jhydrol.2020.125772 (2020).

Xie, K. et al. Physics-guided deep learning for rainfall-runoff modeling by considering extreme events and monotonic relationships. Journal of Hydrology 603, doi:10.1016/j.jhydrol.2021.127043 (2021).

Xie, K. et al. Verification of a New Spatial Distribution Function of Soil Water Storage Capacity Using Conceptual and SWAT Models. Journal of Hydrologic Engineering 25, doi:10.1061/(asce)he.1943-5584.0001887 (2020).


