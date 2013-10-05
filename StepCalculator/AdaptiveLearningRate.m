classdef AdaptiveLearningRate < StepCalculator
   % Implements Tom Schaul and Yann LeCun's "Adaptive learning rates for
   % stochastic, sparse, non-smooth gradients" (2013 arXiv) scheme.
   
   properties
      eps % small value used to prevent division by zero
      C % constant (>= 1) used to initially overestimate gradSquared and hessSquared
         % in order to slow down learning until estimates are accurate.
      n0 % initial memory size
         
      gradAvg % an exponential moving average of past gradient values
      gradSquaredAvg % an exponential moving average of past gradient squared values
      hessAvg % an exponential moving average of finite difference estimates of the diagonal of the Hessian
      hessSquaredAvg % an exponential moving average of finite difference estimates of the square of Hessian diag
      
      learnRates % current per parameter learning rates
      memorySize % current per parameter memories for each parameters moving average
   end
   
   methods
      function obj = AdaptiveLearningRate(varargin)
         p = inputParser;
         p.addParamValue('eps', 1e-5);
         p.addParamValue('C', 2);
         p.addParamValue('n0', 10);
         parse(p, varargin{:});
         
         obj.eps = p.Results.eps;
         obj.C = p.Results.C;
         obj.n0 = p.Results.n0;
      end
      
      function take_step(obj, x, t, model, ~)
         
         if isempty(obj.gradAvg)   
            obj.initialize_averages(x, t, model)
            return;
         end
         
         N = size(x, 2);
         raw_grad1 = model.gradient(x, t, 'averaged', false);
         modelCopy = model.copy();
         modelCopy.increment_params(obj.gradAvg);
         raw_grad2 = modelCopy.gradient(x, t, 'averaged', false);
         clear modelCopy
         step = cell(1, length(raw_grad1));
         
         for i = 1:length(raw_grad1)
            gradDim = ndims(raw_grad1{i});
            grad_diff = bsxfun(@rdivide, abs(raw_grad1{i} - raw_grad2{i}), ...
                              max(abs(obj.gradAvg{i}), obj.eps));
            hess = sum(grad_diff, gradDim)/N; % sum over last dimenion
            grad_diff_squared = grad_diff.^2;
            clear grad_diff
            hessSquared = sum(grad_diff_squared, gradDim)/N;
            clear grad_diff_squared
            raw_grad2{i} = []; % clear room in GPU memory
            grad = sum(raw_grad1{i}, gradDim)/N;
            gradSquared = sum(raw_grad1{i}.^2, gradDim)/N;
            nonZeroTerms = sum(raw_grad1{i}~=0, gradDim);
            raw_grad1{i} = []; % clear room in GPU memory
            
            % Detect outliers and increase memorySize by 1 when detected
            outlierIdx = (abs(grad - obj.gradAvg{i}) > ...
                              2*sqrt(obj.gradSquaredAvg{i} - obj.gradAvg{i}.^2)) | ...
                         (abs(hess - obj.hessAvg{i}) > ...
                              2*sqrt(obj.hessSquaredAvg{i} - obj.hessAvg{i}.^2));
            obj.memorySize{i}(outlierIdx) = obj.memorySize{i}(outlierIdx) + 1;
            
            % Update moving averages
            memInv = 1./obj.memorySize{i};
            obj.gradAvg{i} = (1 - memInv).*obj.gradAvg{i} + grad.*memInv;
            obj.gradSquaredAvg{i} = (1 - memInv).*obj.gradSquaredAvg{i} + gradSquared.*memInv;
            obj.hessAvg{i} = (1 - memInv).*obj.hessAvg{i} + hess.*memInv;
            obj.hessSquaredAvg{i} = (1 - memInv).*obj.hessSquaredAvg{i} + hessSquared.*memInv;   
                              
            % Update Learning Rates
            obj.learnRates{i} = (N*obj.hessAvg{i}.*obj.gradAvg{i}.^2 + obj.eps)./...
                                 (obj.eps + obj.hessSquaredAvg{i}.*...
                                 (obj.gradSquaredAvg{i} + (nonZeroTerms-1).*obj.gradAvg{i}.^2));
                              
            % Update memorySize
            obj.memorySize{i} = obj.memorySize{i}.*(1 - (obj.gradAvg{i}.^2 + obj.eps)./...
                                    (obj.gradSquaredAvg{i} + obj.eps)) + 1;
                              
            % Define model step
            step{i} = -obj.learnRates{i}.*grad;
         end
         
         % Increment model params
         model.increment_params(step);
      end
      
      function initialize_averages(obj, x, t, model)
         N = size(x, 2);
         raw_grad1 = model.gradient(x, t, 'averaged', false);
         gradLength = length(raw_grad1);
         obj.gradAvg = cell(1, gradLength);
         obj.gradSquaredAvg = cell(1, gradLength);
         obj.hessAvg = cell(1, gradLength);
         obj.hessSquaredAvg = cell(1, gradLength);
         
         for i = 1:gradLength
            gradDim = ndims(raw_grad1{i});
            obj.gradAvg{i} = sum(raw_grad1{i}, gradDim)/N;
            obj.gradSquaredAvg{i} = obj.C*sum(raw_grad1{i}.^2, gradDim)/N;
            obj.memorySize{i} = obj.n0*model.gpuState.ones(size(obj.gradAvg{i}));
         end
         
         modelCopy = model.copy();
         modelCopy.increment_params(obj.gradAvg);
         raw_grad2 = modelCopy.gradient(x, t, 'averaged', false);
         clear modelCopy
         
         for i = 1:gradLength
            gradDim = ndims(raw_grad1{i});
            grad_diff = bsxfun(@rdivide, abs(raw_grad1{i} - raw_grad2{i}), ...
                              max(abs(obj.gradAvg{i}), obj.eps));
            obj.hessAvg{i} = sum(grad_diff, gradDim)/N; % sum over last dimenion
            obj.hessSquaredAvg{i} = obj.C*sum(grad_diff.^2, gradDim)/N;
            
            raw_grad2{i} = []; % clear room in GPU memory
            raw_grad1{i} = [];
         end
      end
      
      function reset(obj)
         obj.gradAvg = [];
         obj.gradSquaredAvg = [];
         obj.hessAvg = [];
         obj.hessSquaredAvg = [];
         obj.learnRates = [];
         obj.memorySize = [];
      end
   end
   
end

