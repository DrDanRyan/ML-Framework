function [TP, FP, TN, FN, precision, recall] = compute_confusionmat(y, t)

if any(t < 0) % svm targets
   threshold = 0;
   t(t < 0) = 0; % change to {0, 1} targets
else % logistic targets
   threshold = .5;
end

pred = y > threshold;
TP = sum(pred == 1 & t == 1);
FP = sum(pred == 1 & t == 0);
FN = sum(pred == 0 & t == 1);
TN = sum(pred == 0 & t == 0);

precision = TP/(TP + FP);
recall = TP/(TP + FN);

end

