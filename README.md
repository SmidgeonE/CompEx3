# Computational Simluation of Earth-Moon Orbits


      A Runge-Kutta 4th order algorithm was used to computationally simulate various orbits around the Earth and the Moon. 
      The benefits and shortcomings of this method were discussed. 
      Using data for realistic scenarios of a satellite, a recreation of the Apollo 8 mission was performed, with a resulting period of $T = 230$ hours, similar to the real life period of $T = 136$ hours.


In order to simulate a satellite moving under the influence of external gravitating bodies, the Runge-Kutta 4th order algorithm (RK4) was employed. 
This determined the trajectory computationally, given the acceleration at any time, and initial velocity and displacements. 
Using MatPlotLib's python module, plots of the trajectory for various initial parameters were produced, to investigate the efficacy of the RK4 algorithm. Numpy was utilised due to its ability to efficiently manipulate vectorised data.


# Contact

If the script developed / any of the results interest you, please contact me at oz21652@bristol.ac.uk
