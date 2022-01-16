# SVM From Scratch
This code was written during the writing of my undergraduate thesis as a means to understand the inner details of Support Vector Machines. This includes a rough translation of the original  \epsilon -SVR and \nu -SVR based on the C source code (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)). It also includes an attempt at a re-write of the orignal C++ Online-SVR (http://onlinesvr.altervista.org/). However as of now, some code might be out-dated.

There are two versions of the \epsilon -SVR, one that uses `CVXOPT` (`EpsilonSVR`) and the other with `SMO` (`EpsilonSMO`). Hyperparameters are tuned using `optuna`


Note: This should only be used as a educational tool as it was for me. The orginal `LIBSVM` or `sklearn`  should be used instead for actual cases.

References:

[1] Palar, P.S., Izzaturahman, F., Zuhal, L.R.and Shimoyama, K., Prediction of the Flutter Boundary in Aeroelasticity via a Support Vector Machine.
