EE499 Weekly Activity Sheet

Name: Spencer Aldrich				Week: 5
Fill 1 before the weekly meeting and 2 after the weekly meeting
What did you work on and achieve this week?
Updated Arduino firmware and Python serial logger so that csv’s include proper labels, timestamps, and consistent metadata. 
Reorganized project file structure to separate raw data, software, and other documentation. 
Verified previously collected subway and train data. 
Collected and validated new segments for: 
Car: 2 segments - ~42 minutes total
There are gaps in both segments due to a loss in usb connection while collecting data. Issue has been resolved. 
Walk: 3 segments - ~30 minutes total 



What are you working on/goal for next week?

Complete remaining data collection (by tonight 2/4/25)
Bus data collection (shooting for ~ 4 segments and roughly 1 hour of collection)
Go over all segments in the dataset to make sure that there is correct labeling, balanced data coverage, and consistent formatting. 
Segments should be similar lengths 
Transition from data collection to validation 
Data windowing 
Signal visualization - look for distinguishable patterns between the data I have and compare them with SHL. 
Create a preliminary draft of slides for week 6 presentation.
Visualize data in plots (compare train vs. subway, car vs. bus, etc.) 
Include unique differences between modes of transportation in slides
They should compare features/ should tell us what features are good to train our model and which are not. Break up into plots for accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z and pressure. All modes of transportation should be on one plot. Use legend and other tools. 
Preliminary NN - confusion matrix or accuracy over epochs 

REACH GOAL: 
Train a preliminary NN and see if we can distinguish between the different types of transportations 

