   Effect of different variable values:
   ( reflection part of rubric)
   
   P: Helps to steer the vehicle in accordance to the crosstrack error (cte).
   A higher value of P (e.g. 0.8) creates problems in turns and the car touches
   the yellow line and goes off the road immediately after crossing the bridge.
   A lower value of P (0.01) creates problems on straight road and the car 
   touches the yellow line.
   A value of 0.2 is chosen as an optimal value.
   
   I: Acounts mainly for error accumulated in previous steps and helps steer if
   the accumulated error is high even though the present error is not high. 
   Also helps counter systematic bias, e.g. misaligned wheels. 
   A higher value (0.3) overestimates bias driving the car off the road. A lower value (0)
   creates an offset in car position.
   A value of 0.004 is chosen as optimal.
   
   D: Helps av oid overshooting of the error when cte is accounted for.
   A higher value (5.0) creates wobble. A lower value (2.0) undershoots the error.
   A value of 3.0 is chosen as optimal.
   
   A simple averaging of errors creates a smoother drive.