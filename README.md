MeagerTesting.py - Inspired by Rachel Meager's paper on the metaanalysis of microcredit expansions (./refs/Meager_CreditProgEval.pdf), this repo provides a starting point for implementing models of treatment effects from A/B tests.  

The model in ./src/MeagerTesting.py extends the model in the original paper to jointly estimate effects across all outcomes, times and treatments.

./src/demo_time_course_panel.ipynb contains a demo of how the model works.  It's applied to a biological time course panel containing 32 outcomes, from 62 cell lines, exposed to 4 treatments + control, at 7 non-uniformly distributed time points over an hour.  The treatment outcomes are measured conditional on time, but extensions of the model can similarly condition on other features of the observations.  

Depends on pytorch and pyro

