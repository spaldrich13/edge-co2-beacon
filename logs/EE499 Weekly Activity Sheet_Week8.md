EE499 Weekly Activity Sheet

Name: Spencer Aldrich				Week: 8
Fill 1 before the weekly meeting and 2 after the weekly meeting
What did you work on and achieve this week?
Integrated SHL dataset for coarse pretraining and validated label/feature alignment
Downloaded additional files from multiple users in order to cover all five transportation modes
Mapped SHL columns to the feature channels I am using. 
Found labeling/accuracy issues during early attempts because of mode mismatches. 
Created a balanced SHL pretraining dataset and trained the CNN to high validation accuracy (~97%), and saved pretrained_shl_coarse.keras for transfer
Ran transfer learning (SHL to self data) and found that the setup underperforms
Attempted to freeze conv layers and full fine-tuning with pretrained SHL weights on the self-collected dataset
Observed ~0.80 test accuracy in these transfer runs, which is worse than the self-only baseline (0.83) 
The SHL domain (device placement, sensors, environment) is not aligned enough to transfer cleanly with this architecture and preprocessing. 
Expanded the self-collected dataset by adding 2 more car segments.
Fixed the missing validation coverage for the car class 
Trained and analyzed self-only baseline CNN to identify key failure mode
Found that there is a big confusion between bus and car 
Bus recall was the limiting factor, which means that the model was frequently labeling bus as car.
Val accuracy is substantially lower than train
Improved self-only model architecture to try and distinguish between bus and car more effectively 
Modified Conv2D kernels to span more time steps - done to better show longer motion patterns that distinguish bus vs. car 
Transitioned from 5 second windows to 8 seconds
Result: improved performance and reduced bus/car confusion (95.7%)


What are you working on/goal for next week?

Begin and make substantial progress on the CO2 estimation pipeline 
Conduct tests for untouched specifications (battery, latency, etc.) (For presentation March 7th) 
Create a draft set of slides for final presentation
Return to SHL 
Flip the axis’ three times to see if there is a good orientation match that increases 
