# Differentially Private Spatial Decomposition of 2D Data for Range Queries

## Overview
This study looked at the challenge of producing
accurate answers to range queries on two-dimensional geospatial
data sets while still preserving the privacy of the data set
participants. For verification purposes, the relative errors produced in answering these range queries were compared to those obtained from Zhang et
al. (2016) [1] using the same algorithms on the same data sets.

## Interactive Web-App

Use the [interactive web-app](http://google.com) to see how 
different algorithms spatially decompose 2D geospatial data
to answer range queries that satisfy differential privacy.

## Details of Study 
Read full paper [here](http://google.com)

### 1. Introduction
Data sets that contain the geographic location of individuals
or their activities can be used to enhance business intelligence
and traffic flow. They can also be used in the process of
determining the location and layout of transport systems,
political boundaries, and facilities [2], [3].

The information contained in such data sets are clearly
valuable to researchers. However, publishing data sets like
these can pose a threat to the privacy of the individuals
whose data is contained within them. This study looked at the challenge of publishing geospatial data that can be
used to accurately answer queries for research purposes while
protecting the privacy of the individuals who participated in
the data sets.

More specifically, this study was concerned with producing
accurate answers to query type known as a range query. A
range query is a type of query that returns the amount of data
points contained within a specific region. Range queries have
particular importance when dealing with geospatial data [2].

To address the challenge of preserving both privacy and
data utility, the paradigm of differential privacy can be used.
Differential privacy is a strong privacy guarantee that ensures
the answer to a query has very little difference when applied
to a data set that differs by the participation of any one
individual. This guarantees that no additional information
about an individual can be revealed by their participation in
the data set.

In this study, three differentially private algorithms for answering
range queries are considered and compared. These
are the *Uniform Grid (UG)*, *Simple QuadTree*, and *PrivTree* algorithms as
described in Zhang et al. (2016) [1] and Qardaji et al. (2013) [3]. These algorithms use different spatial
decomposition methods to divide the input data space into
smaller sub-regions before computing a noisy count of data
points contained within each sub-region.

### 2. Differential Privacy
Differential privacy (DP), on an intuitive
level, is a strong privacy guarantee that ensures the answer
to a query has very little difference when applied to a data set
that differs by the participation of any one individual. A more
formal definition is given next.