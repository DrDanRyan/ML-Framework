classdef MomentumSchedule < ParameterSchedule
   % Provides learning rate and momentum parameters to stepCalculators that need
   % them. The learning rate is constant until lrBurnIn updates have passed and
   % then exponentially decays according to lrDecay (which is computed based on
   % a user specified learning rate half-life). The momentum term is ramped up
   % towards is maximum value based on the value C (see properties below).
   
   properties
      params % current values of learning rate and momentum: {lr, momentum}
      
      % initial learning rate; can be single scalar, OR a cell array with length
      % equal to number of layers in the model (in which case gradShape must 
      % also be specified describing the shape of the nested model gradient),
      % OR a cell array of length equal to size of flattened gradient of the 
      % model. 
      lr0 
       
      lrBurnIn % number of updates before lrDecay is applied
      
      % exponential decay rate for learning rate (computed based on user 
      % specified lrHalfLife and applied every update after lrBurnIn)
      lrDecay 
      
      maxMomentum % maximum value for momentum; shape should be the same as lr0
      
      % momentum at update t:= min(maxMomentum, (t + C)/(t + 2*C))
      % for example, t=0 => rho=1/2, t=C => rho=2/3, t=8C => rho=9/10;
      % if C = 0 then momentum = maxMomentum for all updates
      C
      
      % Keep track of number of updates that have passed
      nUpdates = 0 
   end
   
   methods
      function obj = MomentumSchedule(lr0, maxMomentum, varargin) 
         p = inputParser();
         p.addParamValue('lrHalfLife', []);
         p.addParamValue('lrBurnIn', 0);
         p.addParamValue('C', []);
         
         % used to specify lr0 and maxMomentum for each LAYER of network
         p.addParamValue('gradShape', []); 
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
               obj.lr0(startIdx:stopIdx) = mat2cell(lr0{i}*dummy, 1, dummy);
               obj.maxMomentum(startIdx:stopIdx) = ...
                  mat2cell(maxMomentum{i}*dummy, 1, dummy);
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
               obj.params{1} = cellfun(@(lr) obj.lrDecay*lr, obj.params{1}, ...
                  'UniformOutput', false);
            end
            % Set momentum
            if ~isempty(obj.C)
               obj.params{2} = cellfun(@(m) min(m, (obj.nUpdates + obj.C)/...
                  (obj.nUpdates + 2*obj.C)), obj.maxMomentum, ...
                  'UniformOutput', false);
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

