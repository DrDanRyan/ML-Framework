classdef MomentumSchedule < ParameterSchedule
   
   properties
      params % {lr, momentum}
      lr0
      lrDecay % exponential decay rate for learning rate (applied every update after lrBurnIn)
      lrBurnIn % number of updates before lrDecay is applied
      
      % momentum at update t:= min(maxMomentum, (t + C)/(t + 2*C))
      % for example, t=0 => rho=1/2, t=C => rho=2/3, t=8C => rho=9/10
      C
      maxMomentum
      
      % Keep track of number of updates that have passed
      nUpdates = 0 
   end
   
   methods
      function obj = MomentumSchedule(lr0, maxMomentum, varargin) 
         p = inputParser();
         p.addParamValue('lrHalfLife', []);
         p.addParamValue('lrBurnIn', 0);
         p.addParamValue('C', []);
         p.addParamValue('gradShape', []); % used to specify lr0 and maxMomentum for each LAYER of network
         parse(p, varargin{:});
         
         obj.lr0 = lr0;
         obj.maxMomentum = maxMomentum;
         obj.params = {lr0, maxMomentum};
         obj.lrBurnIn = p.Results.lrBurnIn;
         obj.C = p.Results.C;
         gradShape = p.Results.gradShape;
         
         if ~isempty(p.Results.lrHalfLife)
            obj.lrDecay = exp(log(.5)/p.Results.lrHalfLife);
         end
         
         %If parameters are specified layerwise, unroll layerwise specs
         %into componentwise specs using gradShape
         if ~isempty(gradShape)
            startIdx = 1;
            obj.lr0 = cell(1, sum(gradShape));
            obj.maxMomentum = cell(1, sum(gradShape));
            for i = 1:length(gradShape);
               stopIdx = startIdx + gradShape(i) - 1;
               dummy = ones(1, gradShape(i));
               obj.lr0(startIdx:stopIdx) = mat2cell(lr0*dummy, 1, dummy);
               obj.maxMomentum(startIdx:stopIdx) = mat2cell(maxMomentum*dummy, 1, dummy);
               startIdx = stopIdx + 1;
            end
            obj.params = {obj.lr0, obj.maxMomentum};
         end
      end
      
      function params = update(obj)
         obj.nUpdates = obj.nUpdates + 1;

         if isa(obj.lr0, 'cell')
            % Decay learning rate if appropriate
            if ~isempty(obj.lrDecay) && obj.nUpdates > obj.lrBurnIn
               obj.params{1} = cellfun(@(lr) obj.lrDecay*lr, obj.params{1}, 'UniformOutput', false);
            end
            % Set momentum
            if ~isempty(obj.C)
               obj.params{2} = cellfun(@(m) min(m, (obj.nUpdates + obj.C)/(obj.nUpdates + 2*obj.C)), ...
                                          obj.maxMomentum, 'UniformOutput', false);
            end
         else % assume lr0 is a scalar, same params for all of grad
            % Decay learning rate if appropriate
            if ~isempty(obj.lrDecay) && obj.nUpdates > obj.lrBurnIn
               obj.params{1} = obj.lrDecay*obj.params{1};
            end
            % Set momentum
            if ~isempty(obj.C)
               obj.params{2} = min(obj.maxMomentum, ...
                                       (obj.nUpdates + obj.C)/(obj.nUpdates + 2*obj.C));
            end
         end

         params = obj.params;
      end
      
      function reset(obj)
         obj.nUpdates = 0;
         obj.params = {obj.lr0, obj.maxMomentum};
      end
   end
   
end

