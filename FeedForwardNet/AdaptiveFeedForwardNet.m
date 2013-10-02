classdef AdaptiveFeedForwardNet < FeedForwardNet
   % This is an extension of FeedForwardNet that computes two additional
   % terms during the gradient computation: the mean squared gradient and
   % the number of nonzero terms in each component of the minibatch for
   % each component of the gradient. These quantities are required to
   % implemnt Tom Schaul and Yann LeCun's "Adaptive learning rates for
   % stochastic, sparse, non-smooth gradients" (2013 arXiv) scheme.
   %
   % In order to use this model, the HiddenLayer and OutputLayer objects
   % used must be able to respond to backprop_with_variance.
   
   methods
      function obj = AdaptiveFeedForwardNet(varargin)
         obj = FeedForwardNet(varargin{:});
      end
      
      function [grad, output, gradVariance, nonZeroTerms] = gradient(obj, x, t)
         % Computes the gradient for batch input x and target t for all parameters in
         % each hiddenLayer and outputLayer.
         
         nHiddenLayers = length(obj.hiddenLayers);
         y = cell(nHiddenLayers, 1); % output from each hiddenLayer
         dLdy = cell(nHiddenLayers, 1); % derivative of loss function wrt hiddenLayer output
         grad = cell(1, nHiddenLayers+1); % gradient of hiddenLayers and outputLayer (last idx)
         gradVariance = cell(1, nHiddenLayers+1); % mean (over the minibatch) of the gradient squared
         nonZeroTerms = cell(1, nHiddenLayers+1); % number of non-zero terms in minibatch for each gradient component
         
         % feed_forward through hiddenLayers
         if obj.isDropout
            [x, mask] = obj.dropout_mask(x);
         end
         
         y{1} = obj.hiddenLayers{1}.feed_forward(x);
         if obj.isDropout
            y{1} = y{1}.*mask{1};
         end

         for i = 2:nHiddenLayers
            y{i} = obj.hiddenLayers{i}.feed_forward(y{i-1});
            if obj.isDropout
               y{i} = y{i}.*mask{i};
            end
         end
         
         % get outputLayer output and begin backpropagation
         [grad{end}, dLdy{end}, output, gradVariance{end}, nonZeroTerms{end}] = ...
             obj.outputLayer.backprop_with_variance(y{end}, t);
         if obj.isDropout
            dLdy{end} = dLdy{end}.*mask{end};
         end
         
         for i = nHiddenLayers:-1:2
            [grad{i}, dLdy{i-1}, gradVariance{i}, nonZeroTerms{i}] = ...
               obj.hiddenLayers{i}.backprop_with_variance(y{i-1}, y{i}, dLdy{i});
            if obj.isDropout
               dLdy{i-1} = dLdy{i-1}.*mask{i-1};
            end
         end
         [grad{1}, ~, gradVariance{1}, nonZeroTerms{1}] = ...
            obj.hiddenLayers{1}.backprop_with_variance(x, y{1}, dLdy{1});
         grad = obj.unroll_gradient(grad);
         gradVariance = obj.unroll_gradient(gradVariance); % same shape as gradient
         nonZeroTerms = obj.unroll_gradient(nonZeroTerms); % same shape as gradient
      end
   end
   
end

