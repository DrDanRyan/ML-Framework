classdef StandardLayer < handle
   properties
      inputSize
      outputSize
      params % params = {W, b}
      gpuState
      initType
      initScale
      L1Penalty
      L2Penalty   
      isPenalty % boolean indicating if any of the penalty terms are active
      maxFanIn % maximum allowable L2-norm of incoming connection weights to a single output node
      gradType % {averaged, sparse, raw} default is averaged
   end
   
   methods
      function obj = StandardLayer(inputSize, outputSize, varargin)       
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('initType', 'dense');
         p.addParamValue('initScale', []);
         p.addParamValue('L1Penalty', []);
         p.addParamValue('L2Penalty', []);
         p.addParamValue('maxFanIn', []);
         p.addParamValue('gpu', [], @(x) islogical(x));
         p.addParamValue('gradType', 'averaged');
         parse(p, varargin{:});
         
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.gpuState = GPUState(p.Results.gpu);
         
         obj.initType = p.Results.initType;
         obj.initScale = p.Results.initScale;
         obj.init_params();
         
         % Store penalties and maxFanIn constraint
         obj.maxFanIn = p.Results.maxFanIn;
         obj.L1Penalty = p.Results.L1Penalty;
         obj.L2Penalty = p.Results.L2Penalty;
         if (isempty(obj.L1Penalty) && isempty(obj.L2Penalty))
            obj.isPenalty = false;
         else
            obj.isPenalty = true;
         end
         obj.gradType = p.Results.gradType;
      end
      
      function init_params(obj)
         obj.params{1} = matrix_init(obj.outputSize, obj.inputSize, obj.initType, ...
                                          obj.initScale, obj.gpuState);
         obj.params{2} = obj.gpuState.zeros(obj.outputSize, 1);
      end
      
      function gather(obj)
         obj.params{1} = gather(obj.params{1});
         obj.params{2} = gather(obj.params{2});
         obj.gpuState.isGPU = false;
      end
      
      function push_to_GPU(obj)
         obj.params{1} = single(gpuArray(obj.params{1}));
         obj.params{2} = single(gpuArray(obj.params{2}));
         obj.gpuState.isGPU = true;
      end
      
      function increment_params(obj, delta_params)
         obj.params{1} = obj.params{1} + delta_params{1};
         obj.params{2} = obj.params{2} + delta_params{2};
         
         if ~isempty(obj.maxFanIn)
            rowNorms = sqrt(sum(obj.params{1}.^2, 2));
            multiplier = min(1, obj.maxFanIn./rowNorms);
            obj.params{1} = bsxfun(@times, obj.params{1}, multiplier);
         end
      end 
      
      function value = compute_z(obj, x)
         value = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function penalties = compute_penalties(obj)
         penalties = cell(1, 2);
         if (~isempty(obj.L1Penalty) && ~isempty(obj.L2Penalty)) %both penalties active
            penalties{1} = obj.L1Penalty*sign(obj.params{1}) + 2*obj.L2Penalty*obj.params{1};
            penalties{2} = obj.L1Penalty*sign(obj.params{2}) + 2*obj.L2Penalty*obj.params{2};
         elseif (~isempty(obj.L1Penalty) && isempty(obj.L2Penalty)) % L1Penalty only
            penalties{1} = obj.L1Penalty*sign(obj.params{1});
            penalties{2} = obj.L1Penalty*sign(obj.params{2});
         else % L2Penalty only
            penalties{1} = 2*obj.L2Penalty*obj.params{1};
            penalties{2} = 2*obj.L2Penalty*obj.params{2};
         end
      end
      
      function grad = grad_from_dLdz(obj, x, dLdz)
         [L1, N] = size(x);
         L2 = obj.outputSize;
         
         switch obj.gradType
            case 'averaged'
               % Divide sum of gradient terms by batch size.
               grad{1} = dLdz*x'/N;
               grad{2} = mean(dLdz, 2);
            case 'sparse'
               % Divide sum by number of non-zero terms instead of
               % batch size.
               nonZero_dLdz = obj.gpuState.make_numeric(dLdz ~= 0);
               nonZero_xTrans = obj.gpuState.make_numeric(x' ~= 0);
               total_nonZero_dLdw = nonZero_dLdz*nonZero_xTrans;
               total_nonZero_dLdz = sum(nonZero_dLdz, 2);

               total_nonZero_dLdw(total_nonZero_dLdw == 0) = 1; % Prevents dividing by zero below
               total_nonZero_dLdz(total_nonZero_dLdz == 0) = 1; % Prevents dividing by zero below
               grad{1} = dLdz*x'./total_nonZero_dLdw;
               grad{2} = sum(dLdz, 2)./total_nonZero_dLdz;
            case 'raw'
               % Do not consolidate terms at all
               grad{2} = reshape(dLdz, L2, 1, N);
               grad{1} = bsxfun(@times, grad{2}, reshape(x, 1, L1, N));
         end
         
         if obj.isPenalty
            penalties = obj.compute_penalties();
            grad{1} = grad{1} + penalties{1};
            grad{2} = grad{2} + penalties{2};
         end
      end
      
   end
end

