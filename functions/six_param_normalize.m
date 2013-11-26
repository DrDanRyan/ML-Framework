function [xT, params, model] = six_param_normalize(x, nUpdates, clipped, varargin)
% fits the transform xT = a*x + b + c*sqrt(x + d) + e*log(x + f)
% with the criteria that xT minimizes the L2 norm between normcdf and the
% empirical cdf of x (which is the same as empircal cdf of xT since
% transform is monotone)

p = inputParser();
p.addParamValue('lr0', .01);
parse(p, varargin{:});

x = sort(x);
N = length(x);
y = ((1:N) - .5)'/N; % empirical cdf
clippedIdx = y >= .05 & y <= .95;
xClipped = x(clippedIdx);
yClipped = y(clippedIdx);
dm = DataManager();
dm.trainingData = {xClipped, yClipped};
mu = mean(x);
sigma = std(x);
model = NormalizeTransform(mu, sigma, x(1));
trainer = GradientTrainer();
trainer.stepCalculator = IRprop('downFactor', .9);
trainer.parameterSchedule = MomentumSchedule(p.Results.lr0, 0);
trainer.dataManager = dm;
trainer.model = model;

trainer.train(nUpdates);
xT = model.transform(x);
params = model.params;


end

