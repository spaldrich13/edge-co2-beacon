EE499 Weekly Activity Sheet

Name: Spencer Aldrich				Week: 6
Fill 1 before the weekly meeting and 2 after the weekly meeting
What did you work on and achieve this week?
Completed remaining data collection
Bus: 3 segments totaling ~58 minutes
Dataset quality:
Verified labeling, timestamps
Windowed all data into 5-second windows with 50% overlap 
Consistent with literature review 
Applied per-channel z-score normalization using training-set statistics (in google colab with model) 
Generated and analyzed standard deviation plots for accelerometer, gyroscope, and pressure signals
Identified which sensor channels show the strongest separation between transportation modes
Trained a baseline CNN using only self-collected data
Initially observed very high accuracy due to window-level data leakage (99.7% accuracy)
Corrected this by implementing a segment-level train/test split 
Re-trained and evaluated the model - new accuracy is at 83.6%

What are you working on/goal for next week?

Currently developing week 6 progress presentation
Finalize and present on Friday 2/13
Put a slide before slide 8 explaining the model (what layers were used) 
Re-do plots and just plot out raw data. 


Integrate SHL dataset for representation learning and pretraining 
Train model using shl dataset - save model 
Load trained model and continue training with self-collected. Architectures must be same in order for the weights to transfer. 
Fine-tune the pretrained model using self-collected training segments
Evaluate performance on self-collected test segments
Analyze remaining classification confusions 
Begin implementing CO2 estimation pipeline 
Aggregate predicted mode durations 
Apply mode-specific emission factors 
