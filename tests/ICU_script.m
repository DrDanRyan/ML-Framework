clear all
load ICU_SVM_data_gpu
targets(targets == -1) = 0;
[outputs, testLoss] = CV_single_model(inputs, targets, 8, @autoencoder_maxout_ICU_test);
event1 = compute_event1(outputs, targets, true)
lemeshow = compute_Lemeshow(outputs, targets, true)