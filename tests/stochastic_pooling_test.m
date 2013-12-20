layer = StochasticPooling2DLayer(2, 2);
x = rand(1, 1, 4, 4);
x(x<.4) = 0;
x = repmat(x, 1, 10000, 1, 1);
xPool = layer.pool(x, true);
dLdy = ones(size(xPool));
dLdyUnpool = layer.unpool(dLdy);
meanUnpool = mean(dLdyUnpool, 2);