# dipole_microswimmers_magnetic_field

## Abstract

At the microscale, microswimmers have gained attention due to their potential applications in biomedicine, technology, and manufacturing. This thesis presents a computational study of the dynamic and structural properties of a quasi-two-dimensional system of 100 dipolar Janus microswimmers under a constant magnetic field. Using Brownian Dynamics simulations, the influence of the field is analyzed in terms of dynamic and structural properties such as the translational and rotational mean squared displacement (MSD) and the static structure factor. The model includes interactions via Weeks-Chandler-Andersen (WCA) potential and dipolar forces, but neglects hydrodynamic effects, considering that the clusters formed do not reach a sufficient size for such effects to be significant. The validity of the simulation is verified by comparing the MSD in the weak interaction regime with the theoretical solution from the Smoluchowski formalism. Data processing and analysis are performed in Python. The results show that the most efficient dynamics—characterized by greater alignment with the field and reduced aggregation—are achieved with low dipolar moments and greater initial separation. These conditions allow the field to reorient the microswimmers without hindering their self-propulsion in a single direction, enabling effective control of the system with minimal external intervention.

*Keywords*: _microswimmer, Janus particle, Brownian dynamics, magnetic dipole, Smoluchowski equation, field-induced orientation, active matter._

## Repository Contents

* `final_functions.py`: Python module containing the complete set of functions used for data generation, processing, analysis, and visualization related to the Brownian Dynamics simulations.
