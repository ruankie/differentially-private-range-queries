# Differentially Private Spatial Decomposition of 2D Data for Range Queries

## Overview
This study looked at the challenge of producing
accurate answers to range queries on two-dimensional geospatial
data sets while still preserving the privacy of the data set
participants. For verification purposes, the relative errors produced in answering these range queries were compared to those obtained from Zhang et
al. (2016) using the same algorithms on the same data sets.

## Interactive Web-App

Use the [interactive web-app](http://google.com) to see how 
different algorithms spatially decompose 2D geospatial data
to answer range queries that satisfy differential privacy.

## Details of Study 
Read full paper [here](https://github.com/ruankie/differentially-private-range-queries/blob/main/paper.pdf).

The paper contains detailed definitions of differential privacy and
range queries as well as details of all algorithms used, methods
followed, and the results that were obtained.

## References
* *Data Sets:* 
    * [Beijing Taxi Data Set](http://snap.stanford.edu/data/loc-gowalla.html)
    * [Gowalla Data Set](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2F%3Fid%3D152883)
* *Algorithms:*
    * J. Zhang, X. Xiaokui, and X. Xing, ''Privtree: A differentially private algorithm for hierarchical decompositions,''
    In Proceedings of the 2016 International Conference on Management of Data, 2016, pp. 155-170.
    * W. Qardaji, W. Yang and N. Li, ''Differentially private grids for geospatial data,'' 
    2013 IEEE 29th International Conference on Data Engineering (ICDE), Brisbane, QLD, 2013, pp. 757-768, doi: 10.1109/ICDE.2013.6544872.
    * G. Cormode, C. Procopiuc, D. Srivastava, E. Shen and T. Yu, ''Differentially
	Private Spatial Decompositions,'' 2012 IEEE 28th International
	Conference on Data Engineering, Washington, DC, 2012, pp. 20-31, doi:
	10.1109/ICDE.2012.16.