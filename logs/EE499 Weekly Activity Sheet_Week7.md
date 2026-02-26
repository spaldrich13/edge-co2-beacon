EE499 Weekly Activity Sheet

Name: Spencer Aldrich				Week: 7
Fill 1 before the weekly meeting and 2 after the weekly meeting
What did you work on and achieve this week?
Refined how the self-collected data is split:
Implemented new train/validation/test segment-level split
Made sure that there is no data leaking from window to window
Confirmed final baseline accuracy is 80% 
Identified primary failure mode: bus misclassified as car 
Train, subway, and walk classes are all well separated 
Began SHL dataset integration 
Downloaded and analyzed Bag_Motion, Label, and interval label files 



What are you working on/goal for next week?
Find the orientation of the sensor in the backpack by going through two papers 
Need to repeat windowing process - find labels 
Train CNN on SHL 
Save pretrained weights 
Load pretrained weights into 5-class transportation model 
Replace outer layer 
Fine-tune using self-collected data split 
Evaluate on self-collected test split 
Compare performance: 
