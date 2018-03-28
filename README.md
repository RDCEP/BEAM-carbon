# BEAM Carbon

## Installation
1. Clone the repo: `git clone https://github.com/RDCEP/BEAM-carbon.git`.

2. `cd` into the directory.

3. A typical `python setup.py install` should suffice.
  
## Usage
BEAM carbon can be run in a python interpreter or on the command line.

### In python

1. Import the BEAMCarbon object 
   
    ```
    from beam_carbon.beam import BEAMCarbon
    beam = BEAMCarbon()
    ```

2. Set the emissions values

    ```
    beam.emissions = [10., 13., 15., ]
    ```
    or 
    ```
    beam = BEAMCarbon(emissions=[10., 13., 15.])
    ```
    
   The values for emissions must be annualized values. You can use 
   any number of strategies to set the emissions input, but it must be 
   a list or numpy array. If emissions are nor declared, they default
   to one time period with zero emissions. Eg, using pandas, you can load
   the A2 scenario in the input directory:
   
   ```
   import pandas as pd
   import numpy as np
   beam.emissions = np.array(pd.DataFrame.from_csv(
       'input/a2.csv', index_col=1).fillna(0).ix[:, 'emissions'])
   ```
   
   
   
3. Specify the size of the time step and the amount of intervals in each. 
   This determines the number of time BEAM will run each times step. 

    ```
    beam.time_step = 10 
    beam.intervals = 1200
    ```
    or
    ```
    beam = BEAMCarbon(emissions=[10., 13., 15.], time_step=1, intervals=100)
    ```
    
4. Finally, run BEAM.

    ```
    beam.run()
    ```
    
5. Several properties of the BEAMCarbon object can affect its output. You
   can use either a DICE-like or linear temperature model. For more detailed
   runs you can also turn off the temperature-dependent recalibration of k_{1},
   k_{2} and k_{h}. 
   
6. Adjusting the parameters delta and k_{d} can make BEAM more closely match 
   other models. Eg:
   
   ```
   beam.delta = 5
   beam.k_d = .002
   ```
   
### Command line

* `beam_carbon -h` will acquaint you with the basic options.

* Emissions can be specified as a comma-separated list (without any spaces)
  or in a CSV file with no header row or column:
    
    ```
    beam_carbon -e 10,13,15
    beam_carbon --csv "./emissions.csv"
    ```
    
* Output is sent to `stdout` but cen be directed to a CSV file instead:
 
    ```
    beam_carbon -e 10,13,15 -o "./beam_output.csv"
    ```
    
* As in python, the emissions time step and BEAM interval can be specified:

    ```
    beam_carbon -e 10,13,15 --timestep 10 --interval 1200
    ```

* Delta and k_{d} can also be specified on the command-line:
    
    ```
    beam_Carbon -e 10,13,15 --timestep 10 --interval 1200 --delta 5 --kd .002
    ```

## License

This code is distributed under the Apache 2 License.