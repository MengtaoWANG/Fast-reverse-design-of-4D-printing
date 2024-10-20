1

   Abaqus 2023.HF2                                  Date 27-10ÔÂ-2023   Time 16:29:05
   For use by RITSUMEI under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     C O U P L E D - T E M P E R A T U R E - D I S P L A C E M E N T   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-02
          AND A TOTAL TIME PERIOD OF                            1.00    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-06
          THE MAXIMUM TIME INCREMENT ALLOWED IS                 1.00    

          CREEP AND SWELLING EFFECTS ARE OMITTED IN THIS STEP

     THE SIZE OF THE TIME INCREMENT IS CONTROLLED BY -
          THE TEMPERATURE CHANGE PER INCREMENT NOT EXCEEDING            15.0    
          THE RATE DEPENDENCY WILL BE CONSIDERED IN THIS STEP

          CREEP AND SWELLING EFFECTS WILL BE IGNORED.

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR HEAT FLUX
     CRITERION FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR TEMP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE HEAT FLUX                           1.000E-02
     AVERAGE HEAT FLUX IS TIME AVERAGE HEAT FLUX
     ALTERNATE CRIT. FOR RESIDUAL HEAT FLUX FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO HEAT FLUX RELATIVE TO TIME AVRG. HEAT FLUX     1.000E-05
     CRITERION FOR RESIDUAL HEAT FLUX WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR TEMP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL HEAT FLUX FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO HEAT FLUX REL. TO TIME AVRG. MAX. HEAT FLUX    1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CONSECUTIVE INCS MEETING TIME INTEG. TOL. FOR TIME INC. INCREASE          3
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     CUT-BACK FACTOR WHEN TIME INTEGRATION ACCURACY TOL. EXCEEDED     0.8500    
     MAX. RATIO OF TIME INTEG. MEASURE TO TOL. FOR TIME INC. INCREASE 0.7500    
     INCREASE FACTOR FOR TIME INCREMENT WHEN POSSIBLE                 0.8000    
     MIN. TIME INC. INCREASE FACTOR IN LINEAR TRANSIENT PROBLEMS      0.9500    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                     100

          LARGE DISPLACEMENT THEORY WILL BE USED

          UNSYMMETRIC MATRIX STORAGE AND SOLUTION WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     0.944    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     ELEMENT OPERATIONS WILL BE CARRIED OUT IN PARALLEL USING   4 THREADS ON 1 DOMAIN

     PARALLEL CONTACT TRACKING ENABLED


     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS
 
     GPU SOLVER ACCELERATION ENABLED



  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
     INITIAL ELEMENT LOAD BALANCE
            DOMAIN    1     THREAD    NUMBER OF ELEMENTS
                              1            752
                              2            752
                              3            736
                              4            760
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  325ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.041E-05   TIME AVG. FORCE       4.041E-05
 LARGEST SCALED RESIDUAL FORCE     -5.197E-04   AT NODE       5167   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -5.197E-04
 LARGEST INCREMENT OF DISP.         -2.06       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        -2.06       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.         1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  314ms
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.199E-05   TIME AVG. FORCE       2.199E-05
 LARGEST SCALED RESIDUAL FORCE     -1.203E-06   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.203E-06
 LARGEST INCREMENT OF DISP.         -2.06       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        2.752E-02   AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2891   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  269ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.199E-05   TIME AVG. FORCE       2.199E-05
 LARGEST SCALED RESIDUAL FORCE      1.513E-07   AT NODE      13091   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.513E-07
 LARGEST INCREMENT OF DISP.         -2.06       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        8.272E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4085   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  300ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.199E-05   TIME AVG. FORCE       2.199E-05
 LARGEST SCALED RESIDUAL FORCE     -3.681E-09   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -3.681E-09
 LARGEST INCREMENT OF DISP.         -2.06       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        1.008E-05   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4049   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  1.000E-02
 STEP TIME COMPLETED       1.000E-02,  TOTAL TIME COMPLETED        1.000E-02


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  301ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.643E-05   TIME AVG. FORCE       3.421E-05
 LARGEST SCALED RESIDUAL FORCE     -2.791E-06   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.791E-06
 LARGEST INCREMENT OF DISP.         -2.05       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        7.282E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3469   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  316ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.130E-05   TIME AVG. FORCE       3.665E-05
 LARGEST SCALED RESIDUAL FORCE     -4.183E-08   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -4.183E-08
 LARGEST INCREMENT OF DISP.         -2.05       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -2.781E-04   AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4037   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  2.000E-02
 STEP TIME COMPLETED       2.000E-02,  TOTAL TIME COMPLETED        2.000E-02


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  297ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.266E-05   TIME AVG. FORCE       4.865E-05
 LARGEST SCALED RESIDUAL FORCE     -2.911E-06   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.911E-06
 LARGEST INCREMENT OF DISP.          2.11       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        7.354E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3639   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  322ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.025E-05   TIME AVG. FORCE       5.118E-05
 LARGEST SCALED RESIDUAL FORCE     -4.357E-08   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -4.357E-08
 LARGEST INCREMENT OF DISP.          2.11       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -3.970E-04   AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.00       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3751   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    
          TIME INCREMENT MAY NOW INCREASE TO    1.500E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  3.000E-02
 STEP TIME COMPLETED       3.000E-02,  TOTAL TIME COMPLETED        3.000E-02


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.500E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  309ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.156E-04   TIME AVG. FORCE       6.728E-05
 LARGEST SCALED RESIDUAL FORCE     -1.111E-05   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.111E-05
 LARGEST INCREMENT OF DISP.          3.30       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.140       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.50       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3467   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  296ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.278E-04   TIME AVG. FORCE       7.034E-05
 LARGEST SCALED RESIDUAL FORCE     -1.971E-07   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.971E-07
 LARGEST INCREMENT OF DISP.          3.30       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.537E-03   AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          1.50       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3783   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    
          TIME INCREMENT MAY NOW INCREASE TO    2.250E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.500E-02,  FRACTION OF STEP COMPLETED  4.500E-02
 STEP TIME COMPLETED       4.500E-02,  TOTAL TIME COMPLETED        4.500E-02


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.250E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  293ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.904E-04   TIME AVG. FORCE       9.436E-05
 LARGEST SCALED RESIDUAL FORCE     -5.374E-05   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -5.374E-05
 LARGEST INCREMENT OF DISP.          5.25       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.322       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          2.25       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4095   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  288ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.031E-04   TIME AVG. FORCE       9.689E-05
 LARGEST SCALED RESIDUAL FORCE     -1.102E-06   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.102E-06
 LARGEST INCREMENT OF DISP.          5.24       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -7.149E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          2.25       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       1429   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  289ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.111E-04   TIME AVG. FORCE       9.849E-05
 LARGEST SCALED RESIDUAL FORCE      8.492E-08   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      8.492E-08
 LARGEST INCREMENT OF DISP.          5.24       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -8.567E-04   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          2.25       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3883   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    
          TIME INCREMENT MAY NOW INCREASE TO    3.375E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.250E-02,  FRACTION OF STEP COMPLETED  6.750E-02
 STEP TIME COMPLETED       6.750E-02,  TOTAL TIME COMPLETED        6.750E-02


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.375E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  283ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.316E-04   TIME AVG. FORCE       1.373E-04
 LARGEST SCALED RESIDUAL FORCE     -2.764E-04   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.764E-04
 LARGEST INCREMENT OF DISP.          8.57       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.757       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.38       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -4.441E-16   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  314ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.352E-04   TIME AVG. FORCE       1.379E-04
 LARGEST SCALED RESIDUAL FORCE     -6.361E-06   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -6.361E-06
 LARGEST INCREMENT OF DISP.          8.54       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -2.578E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.38       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3643   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  298ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.366E-04   TIME AVG. FORCE       1.382E-04
 LARGEST SCALED RESIDUAL FORCE      2.087E-06   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.087E-06
 LARGEST INCREMENT OF DISP.          8.53       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.598E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.38       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3625   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  308ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.664E-04   TIME AVG. FORCE       1.431E-04
 LARGEST SCALED RESIDUAL FORCE      7.076E-08   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      7.076E-08
 LARGEST INCREMENT OF DISP.          8.53       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        1.313E-04   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.38       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       1423   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    
          TIME INCREMENT MAY NOW INCREASE TO    5.063E-02

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.375E-02,  FRACTION OF STEP COMPLETED  0.101    
 STEP TIME COMPLETED       0.101    ,  TOTAL TIME COMPLETED        0.101    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  308ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.284E-04   TIME AVG. FORCE       2.125E-04
 LARGEST SCALED RESIDUAL FORCE      1.626E-03   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.626E-03
 LARGEST INCREMENT OF DISP.          14.5       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         1.85       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        1.776E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  302ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.260E-04   TIME AVG. FORCE       2.121E-04
 LARGEST SCALED RESIDUAL FORCE     -3.906E-05   AT NODE       3936   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -3.906E-05
 LARGEST INCREMENT OF DISP.          14.4       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -6.317E-02   AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3741   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  273ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.239E-04   TIME AVG. FORCE       2.118E-04
 LARGEST SCALED RESIDUAL FORCE      4.230E-05   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.230E-05
 LARGEST INCREMENT OF DISP.          14.3       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.166       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3339   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  323ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.355E-04   TIME AVG. FORCE       2.135E-04
 LARGEST SCALED RESIDUAL FORCE      2.323E-06   AT NODE      12975   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.323E-06
 LARGEST INCREMENT OF DISP.          14.3       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        6.415E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3793   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  292ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.847E-04   TIME AVG. FORCE       2.205E-04
 LARGEST SCALED RESIDUAL FORCE     -1.832E-07   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.832E-07
 LARGEST INCREMENT OF DISP.          14.3       AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.906E-04   AT NODE      13503   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       1037   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.152    
 STEP TIME COMPLETED       0.152    ,  TOTAL TIME COMPLETED        0.152    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  293ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.047E-03   TIME AVG. FORCE       3.239E-04
 LARGEST SCALED RESIDUAL FORCE      2.504E-03   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.504E-03
 LARGEST INCREMENT OF DISP.          16.5       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         2.34       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -3.553E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  274ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.043E-03   TIME AVG. FORCE       3.233E-04
 LARGEST SCALED RESIDUAL FORCE      5.363E-05   AT NODE      13169   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      5.363E-05
 LARGEST INCREMENT OF DISP.          16.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -9.176E-02   AT NODE         62   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3223   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  329ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.038E-03   TIME AVG. FORCE       3.227E-04
 LARGEST SCALED RESIDUAL FORCE      7.693E-05   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      7.693E-05
 LARGEST INCREMENT OF DISP.          16.1       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.311       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3485   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  288ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.055E-03   TIME AVG. FORCE       3.249E-04
 LARGEST SCALED RESIDUAL FORCE      5.650E-06   AT NODE      13146   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      5.650E-06
 LARGEST INCREMENT OF DISP.          16.1       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        1.601E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2885   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  308ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.131E-03   TIME AVG. FORCE       3.343E-04
 LARGEST SCALED RESIDUAL FORCE     -5.250E-07   AT NODE      12978   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -5.250E-07
 LARGEST INCREMENT OF DISP.          16.1       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.583E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2907   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.203    
 STEP TIME COMPLETED       0.203    ,  TOTAL TIME COMPLETED        0.203    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  301ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.623E-03   TIME AVG. FORCE       4.774E-04
 LARGEST SCALED RESIDUAL FORCE      2.813E-03   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.813E-03
 LARGEST INCREMENT OF DISP.          18.5       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         2.45       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        5.329E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  287ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.622E-03   TIME AVG. FORCE       4.774E-04
 LARGEST SCALED RESIDUAL FORCE      6.232E-05   AT NODE      13169   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      6.232E-05
 LARGEST INCREMENT OF DISP.          18.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.118       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2605   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  291ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.614E-03   TIME AVG. FORCE       4.765E-04
 LARGEST SCALED RESIDUAL FORCE      1.038E-04   AT NODE      12512   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.038E-04
 LARGEST INCREMENT OF DISP.          18.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.419       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4201   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  289ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.647E-03   TIME AVG. FORCE       4.802E-04
 LARGEST SCALED RESIDUAL FORCE      1.036E-05   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.036E-05
 LARGEST INCREMENT OF DISP.          18.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        2.598E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3493   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  299ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.712E-03   TIME AVG. FORCE       4.874E-04
 LARGEST SCALED RESIDUAL FORCE     -1.155E-06   AT NODE      12978   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.155E-06
 LARGEST INCREMENT OF DISP.          18.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -3.501E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3053   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.253    
 STEP TIME COMPLETED       0.253    ,  TOTAL TIME COMPLETED        0.253    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  285ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.408E-03   TIME AVG. FORCE       6.795E-04
 LARGEST SCALED RESIDUAL FORCE      3.894E-03   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      3.894E-03
 LARGEST INCREMENT OF DISP.          20.5       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         2.53       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3655   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  274ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.411E-03   TIME AVG. FORCE       6.798E-04
 LARGEST SCALED RESIDUAL FORCE      7.029E-05   AT NODE      13169   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      7.029E-05
 LARGEST INCREMENT OF DISP.          20.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.145       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3469   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  294ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.401E-03   TIME AVG. FORCE       6.787E-04
 LARGEST SCALED RESIDUAL FORCE     -1.566E-04   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.566E-04
 LARGEST INCREMENT OF DISP.          19.9       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.536       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3937   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  309ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.434E-03   TIME AVG. FORCE       6.820E-04
 LARGEST SCALED RESIDUAL FORCE      2.019E-05   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.019E-05
 LARGEST INCREMENT OF DISP.          19.9       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        3.868E-02   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4201   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  272ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.522E-03   TIME AVG. FORCE       6.909E-04
 LARGEST SCALED RESIDUAL FORCE     -2.679E-06   AT NODE      12978   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.679E-06
 LARGEST INCREMENT OF DISP.          19.9       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -6.184E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3923   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.304    
 STEP TIME COMPLETED       0.304    ,  TOTAL TIME COMPLETED        0.304    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  310ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.463E-03   TIME AVG. FORCE       9.429E-04
 LARGEST SCALED RESIDUAL FORCE      5.487E-03   AT NODE      12978   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      5.487E-03
 LARGEST INCREMENT OF DISP.          22.5       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         2.58       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        3.553E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  299ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.467E-03   TIME AVG. FORCE       9.433E-04
 LARGEST SCALED RESIDUAL FORCE     -9.505E-05   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -9.505E-05
 LARGEST INCREMENT OF DISP.          22.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.173       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3467   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  303ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.465E-03   TIME AVG. FORCE       9.430E-04
 LARGEST SCALED RESIDUAL FORCE     -2.221E-04   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.221E-04
 LARGEST INCREMENT OF DISP.          21.6       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.697       AT NODE         64   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3661   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  296ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.482E-03   TIME AVG. FORCE       9.446E-04
 LARGEST SCALED RESIDUAL FORCE      3.665E-05   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      3.665E-05
 LARGEST INCREMENT OF DISP.          21.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        5.435E-02   AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3743   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  306ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.585E-03   TIME AVG. FORCE       9.540E-04
 LARGEST SCALED RESIDUAL FORCE     -5.385E-06   AT NODE      12978   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -5.385E-06
 LARGEST INCREMENT OF DISP.          21.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -9.155E-03   AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3907   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  295ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      3.790E-03   TIME AVG. FORCE       9.726E-04
 LARGEST SCALED RESIDUAL FORCE     -2.380E-07   AT NODE      13081   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.380E-07
 LARGEST INCREMENT OF DISP.          21.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -4.553E-04   AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4037   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.354    
 STEP TIME COMPLETED       0.354    ,  TOTAL TIME COMPLETED        0.354    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  321ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.252E-03   TIME AVG. FORCE       1.246E-03
 LARGEST SCALED RESIDUAL FORCE      1.301E-02   AT NODE      13172   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.301E-02
 LARGEST INCREMENT OF DISP.          24.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         4.30       AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -7.105E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  274ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.197E-03   TIME AVG. FORCE       1.241E-03
 LARGEST SCALED RESIDUAL FORCE     -7.978E-04   AT NODE      12512   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -7.978E-04
 LARGEST INCREMENT OF DISP.          24.9       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.799       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3343   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  333ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.242E-03   TIME AVG. FORCE       1.245E-03
 LARGEST SCALED RESIDUAL FORCE      8.890E-03   AT NODE      13164   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      8.890E-03
 LARGEST INCREMENT OF DISP.          28.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         4.75       AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4191   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  304ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.208E-03   TIME AVG. FORCE       1.242E-03
 LARGEST SCALED RESIDUAL FORCE     -2.241E-04   AT NODE      12512   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.241E-04
 LARGEST INCREMENT OF DISP.          28.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.315       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3035   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  323ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.199E-03   TIME AVG. FORCE       1.241E-03
 LARGEST SCALED RESIDUAL FORCE      2.232E-03   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.232E-03
 LARGEST INCREMENT OF DISP.          29.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         1.87       AT NODE         64   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3323   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  290ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.221E-03   TIME AVG. FORCE       1.243E-03
 LARGEST SCALED RESIDUAL FORCE     -7.954E-05   AT NODE      12622   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -7.954E-05
 LARGEST INCREMENT OF DISP.          29.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        8.938E-02   AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3311   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  331ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      4.201E-03   TIME AVG. FORCE       1.242E-03
 LARGEST SCALED RESIDUAL FORCE      9.760E-05   AT NODE      13164   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      9.760E-05
 LARGEST INCREMENT OF DISP.          29.8       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.132       AT NODE         64   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3033   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  282ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                      4.602E-03   TIME AVG. FORCE       1.275E-03
 LARGEST SCALED RESIDUAL FORCE     -3.384E-06   AT NODE      12590   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -3.384E-06
 LARGEST INCREMENT OF DISP.          29.8       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        3.914E-03   AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3457   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   8 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  8 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.405    
 STEP TIME COMPLETED       0.405    ,  TOTAL TIME COMPLETED        0.405    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  311ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.271E-03   TIME AVG. FORCE       1.505E-03
 LARGEST SCALED RESIDUAL FORCE      4.879E-03   AT NODE      13172   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.879E-03
 LARGEST INCREMENT OF DISP.          25.6       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        -4.24       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3741   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  300ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.292E-03   TIME AVG. FORCE       1.507E-03
 LARGEST SCALED RESIDUAL FORCE     -1.344E-04   AT NODE      12985   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.344E-04
 LARGEST INCREMENT OF DISP.          25.6       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.100       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2403   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  319ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.279E-03   TIME AVG. FORCE       1.506E-03
 LARGEST SCALED RESIDUAL FORCE      1.395E-04   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.395E-04
 LARGEST INCREMENT OF DISP.          25.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.242       AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3053   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  287ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.625E-03   TIME AVG. FORCE       1.533E-03
 LARGEST SCALED RESIDUAL FORCE      6.903E-06   AT NODE      12985   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      6.903E-06
 LARGEST INCREMENT OF DISP.          25.7       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        6.210E-03   AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4069   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.456    
 STEP TIME COMPLETED       0.456    ,  TOTAL TIME COMPLETED        0.456    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.063E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  326ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.247E-03   TIME AVG. FORCE       1.727E-03
 LARGEST SCALED RESIDUAL FORCE      4.790E-03   AT NODE      13082   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.790E-03
 LARGEST INCREMENT OF DISP.          26.4       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         3.78       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        7.105E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  282ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.316E-03   TIME AVG. FORCE       1.731E-03
 LARGEST SCALED RESIDUAL FORCE     -1.569E-04   AT NODE      12512   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.569E-04
 LARGEST INCREMENT OF DISP.          26.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.114       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2409   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  311ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.371E-03   TIME AVG. FORCE       1.735E-03
 LARGEST SCALED RESIDUAL FORCE      2.047E-05   AT NODE      13344   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.047E-05
 LARGEST INCREMENT OF DISP.          26.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        4.897E-02   AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3209   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  291ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.694E-03   TIME AVG. FORCE       1.758E-03
 LARGEST SCALED RESIDUAL FORCE      2.357E-06   AT NODE      12732   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.357E-06
 LARGEST INCREMENT OF DISP.          26.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.781E-03   AT NODE         62   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          5.06       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3789   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    
          TIME INCREMENT MAY NOW INCREASE TO    7.594E-02

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.063E-02,  FRACTION OF STEP COMPLETED  0.506    
 STEP TIME COMPLETED       0.506    ,  TOTAL TIME COMPLETED        0.506    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  296ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.266E-03   TIME AVG. FORCE       1.926E-03
 LARGEST SCALED RESIDUAL FORCE      1.536E-02   AT NODE      13172   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.536E-02
 LARGEST INCREMENT OF DISP.          39.5       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         8.29       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -1.066E-14   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  284ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.195E-03   TIME AVG. FORCE       1.921E-03
 LARGEST SCALED RESIDUAL FORCE     -4.889E-04   AT NODE      12512   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -4.889E-04
 LARGEST INCREMENT OF DISP.          39.1       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.471       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3049   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  277ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.193E-03   TIME AVG. FORCE       1.921E-03
 LARGEST SCALED RESIDUAL FORCE      1.967E-04   AT NODE      12622   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.967E-04
 LARGEST INCREMENT OF DISP.          39.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.352       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3085   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  308ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.242E-03   TIME AVG. FORCE       1.924E-03
 LARGEST SCALED RESIDUAL FORCE      2.683E-05   AT NODE      13272   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.683E-05
 LARGEST INCREMENT OF DISP.          39.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        4.328E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4223   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  283ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.559E-03   TIME AVG. FORCE       1.945E-03
 LARGEST SCALED RESIDUAL FORCE      5.987E-06   AT NODE      13081   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      5.987E-06
 LARGEST INCREMENT OF DISP.          39.0       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -7.314E-03   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3763   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.582    
 STEP TIME COMPLETED       0.582    ,  TOTAL TIME COMPLETED        0.582    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  289ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.189E-03   TIME AVG. FORCE       2.085E-03
 LARGEST SCALED RESIDUAL FORCE      1.565E-02   AT NODE      13082   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.565E-02
 LARGEST INCREMENT OF DISP.          36.9       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         10.8       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -7.105E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  300ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.089E-03   TIME AVG. FORCE       2.079E-03
 LARGEST SCALED RESIDUAL FORCE     -4.376E-04   AT NODE      13081   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -4.376E-04
 LARGEST INCREMENT OF DISP.          36.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.687       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3073   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  295ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.084E-03   TIME AVG. FORCE       2.079E-03
 LARGEST SCALED RESIDUAL FORCE      2.454E-04   AT NODE      12622   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.454E-04
 LARGEST INCREMENT OF DISP.          36.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.690       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4055   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  288ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.137E-03   TIME AVG. FORCE       2.082E-03
 LARGEST SCALED RESIDUAL FORCE      5.904E-05   AT NODE      12858   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      5.904E-05
 LARGEST INCREMENT OF DISP.          36.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.149       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4177   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  298ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.243E-03   TIME AVG. FORCE       2.089E-03
 LARGEST SCALED RESIDUAL FORCE     -1.860E-05   AT NODE      12978   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.860E-05
 LARGEST INCREMENT OF DISP.          36.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -4.710E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       1423   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  281ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.489E-03   TIME AVG. FORCE       2.104E-03
 LARGEST SCALED RESIDUAL FORCE     -2.198E-06   AT NODE      13181   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.198E-06
 LARGEST INCREMENT OF DISP.          36.3       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -5.696E-03   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4051   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.658    
 STEP TIME COMPLETED       0.658    ,  TOTAL TIME COMPLETED        0.658    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  284ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.016E-03   TIME AVG. FORCE       2.217E-03
 LARGEST SCALED RESIDUAL FORCE     -1.125E-02   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.125E-02
 LARGEST INCREMENT OF DISP.          42.5       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         11.0       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        7.105E-15   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  325ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.950E-03   TIME AVG. FORCE       2.213E-03
 LARGEST SCALED RESIDUAL FORCE      4.471E-04   AT NODE      12622   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.471E-04
 LARGEST INCREMENT OF DISP.          42.2       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.719       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4183   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  291ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.971E-03   TIME AVG. FORCE       2.214E-03
 LARGEST SCALED RESIDUAL FORCE      1.754E-04   AT NODE      12622   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.754E-04
 LARGEST INCREMENT OF DISP.          41.5       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.636       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4241   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  300ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.033E-03   TIME AVG. FORCE       2.218E-03
 LARGEST SCALED RESIDUAL FORCE      3.476E-05   AT NODE      13091   DOF  3
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      3.476E-05
 LARGEST INCREMENT OF DISP.          41.6       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.118       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3079   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  293ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.126E-03   TIME AVG. FORCE       2.223E-03
 LARGEST SCALED RESIDUAL FORCE      1.031E-05   AT NODE      13173   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.031E-05
 LARGEST INCREMENT OF DISP.          41.6       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -3.264E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4229   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.734    
 STEP TIME COMPLETED       0.734    ,  TOTAL TIME COMPLETED        0.734    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  299ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.820E-03   TIME AVG. FORCE       2.312E-03
 LARGEST SCALED RESIDUAL FORCE     -1.212E-02   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.212E-02
 LARGEST INCREMENT OF DISP.          52.1       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.         10.5       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2405   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  314ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.753E-03   TIME AVG. FORCE       2.308E-03
 LARGEST SCALED RESIDUAL FORCE      4.632E-04   AT NODE      12622   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.632E-04
 LARGEST INCREMENT OF DISP.          51.5       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.721       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4195   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  281ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.763E-03   TIME AVG. FORCE       2.309E-03
 LARGEST SCALED RESIDUAL FORCE      2.042E-04   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.042E-04
 LARGEST INCREMENT OF DISP.          50.6       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.902       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4051   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  314ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.773E-03   TIME AVG. FORCE       2.309E-03
 LARGEST SCALED RESIDUAL FORCE     -6.600E-05   AT NODE      13344   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -6.600E-05
 LARGEST INCREMENT OF DISP.          50.9       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        0.235       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4179   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  286ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.879E-03   TIME AVG. FORCE       2.315E-03
 LARGEST SCALED RESIDUAL FORCE      2.333E-05   AT NODE      13344   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      2.333E-05
 LARGEST INCREMENT OF DISP.          50.8       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -8.645E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3193   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  331ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.101E-03   TIME AVG. FORCE       2.327E-03
 LARGEST SCALED RESIDUAL FORCE      3.317E-06   AT NODE      13074   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      3.317E-06
 LARGEST INCREMENT OF DISP.          50.8       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.342E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4241   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.810    
 STEP TIME COMPLETED       0.810    ,  TOTAL TIME COMPLETED        0.810    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  285ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.580E-03   TIME AVG. FORCE       2.393E-03
 LARGEST SCALED RESIDUAL FORCE     -1.136E-02   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.136E-02
 LARGEST INCREMENT OF DISP.          59.2       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        -12.5       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3747   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  318ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.535E-03   TIME AVG. FORCE       2.391E-03
 LARGEST SCALED RESIDUAL FORCE      4.436E-04   AT NODE      12622   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.436E-04
 LARGEST INCREMENT OF DISP.          58.4       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.755       AT NODE         64   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4191   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  298ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.542E-03   TIME AVG. FORCE       2.391E-03
 LARGEST SCALED RESIDUAL FORCE      1.165E-04   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.165E-04
 LARGEST INCREMENT OF DISP.          57.8       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.592       AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3785   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  323ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.591E-03   TIME AVG. FORCE       2.394E-03
 LARGEST SCALED RESIDUAL FORCE     -2.426E-05   AT NODE      13344   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.426E-05
 LARGEST INCREMENT OF DISP.          57.9       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        9.587E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4055   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  316ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.818E-03   TIME AVG. FORCE       2.406E-03
 LARGEST SCALED RESIDUAL FORCE      7.357E-06   AT NODE      13344   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      7.357E-06
 LARGEST INCREMENT OF DISP.          57.9       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -2.458E-02   AT NODE         65   DOF  3
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       2011   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.886    
 STEP TIME COMPLETED       0.886    ,  TOTAL TIME COMPLETED        0.886    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  290ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.322E-03   TIME AVG. FORCE       2.452E-03
 LARGEST SCALED RESIDUAL FORCE     -1.020E-02   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.020E-02
 LARGEST INCREMENT OF DISP.          63.2       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        -16.2       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4053   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  297ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.305E-03   TIME AVG. FORCE       2.451E-03
 LARGEST SCALED RESIDUAL FORCE      4.006E-04   AT NODE      12622   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.006E-04
 LARGEST INCREMENT OF DISP.          62.2       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.977       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3057   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  291ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.308E-03   TIME AVG. FORCE       2.451E-03
 LARGEST SCALED RESIDUAL FORCE      1.032E-04   AT NODE      13172   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.032E-04
 LARGEST INCREMENT OF DISP.          61.6       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.621       AT NODE      13073   DOF  3
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3047   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  307ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.381E-03   TIME AVG. FORCE       2.455E-03
 LARGEST SCALED RESIDUAL FORCE     -2.476E-05   AT NODE      13146   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -2.476E-05
 LARGEST INCREMENT OF DISP.          61.7       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.124       AT NODE         64   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4077   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  306ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.540E-03   TIME AVG. FORCE       2.463E-03
 LARGEST SCALED RESIDUAL FORCE      8.129E-06   AT NODE      13344   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      8.129E-06
 LARGEST INCREMENT OF DISP.          61.7       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        3.471E-02   AT NODE         64   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          7.59       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3767   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-02,  FRACTION OF STEP COMPLETED  0.962    
 STEP TIME COMPLETED       0.962    ,  TOTAL TIME COMPLETED        0.962    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.812E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  291ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.152E-03   TIME AVG. FORCE       2.495E-03
 LARGEST SCALED RESIDUAL FORCE     -1.130E-03   AT NODE      13082   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.130E-03
 LARGEST INCREMENT OF DISP.          31.1       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.        -6.47       AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.81       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.        4.441E-16   AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  303ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.503E-03   TIME AVG. FORCE       2.512E-03
 LARGEST SCALED RESIDUAL FORCE      4.900E-05   AT NODE      12622   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      4.900E-05
 LARGEST INCREMENT OF DISP.          30.9       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.210       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.81       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3031   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  289ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.328E-03   TIME AVG. FORCE       2.504E-03
 LARGEST SCALED RESIDUAL FORCE     -1.477E-05   AT NODE      13164   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE     -1.477E-05
 LARGEST INCREMENT OF DISP.          31.1       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -0.343       AT NODE         65   DOF  2
   INSTANCE: PART-21-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.81       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       3741   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST x 1 MPI RANK PER HOST x (4 THREADS PER RANK + 1 GPU PER RANK)
        NUMBER OF EQUATIONS:  51081
        NUMBER OF RHS:        1
        NUMBER OF FLOPS:      3.568e+10
        SOLVER ELAPSED TIME:  301ms

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.516E-03   TIME AVG. FORCE       2.513E-03
 LARGEST SCALED RESIDUAL FORCE      1.059E-06   AT NODE       3918   DOF  2
   INSTANCE: PART-21-1                                                                       
  CORRESPONDING RESIDUAL FORCE      1.059E-06
 LARGEST INCREMENT OF DISP.          31.1       AT NODE         66   DOF  3
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO DISP.       -2.308E-03   AT NODE         66   DOF  2
   INSTANCE: PART-21-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE HEAT FLUX                   9.33       TIME AVG. HEAT FLUX    9.33    
 ALL HEAT FLUX RESIDUALS ARE ZERO
 LARGEST INCREMENT OF TEMP.          3.81       AT NODE          1   DOF 11
   INSTANCE: PART-21-1                                                                       
 LARGEST CORRECTION TO TEMP.       -6.533E-35   AT NODE       4093   DOF 11
   INSTANCE: PART-21-1                                                                       
          THE HEAT FLUX EQUILIBRIUM EQUATIONS HAVE CONVERGED

 MAXIMUM NON-PRESCRIBED INCREMENT IN TEMPERATURE IS    0.00    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.812E-02,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        1.00    ,  TOTAL TIME COMPLETED         1.00    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         21  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                      95  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                      95  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      95  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       2  WARNING MESSAGES DURING USER INPUT PROCESSING
                       1  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       1  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =     1.47E+02
       SYSTEM TIME (SEC)    =      6.0    
       TOTAL CPU TIME (SEC) =     1.53E+02
       WALLCLOCK TIME (SEC) =           51
       MEMORY PEAK (GB)     =            1
