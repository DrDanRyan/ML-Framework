classdef MomentumSchedule < ParameterSchedule
   
   properties
      params % {lr, momentum}
      lr0
      lrDecay % exponential decay rate for learning rate
      lrBurnIn % number of updates before lrDecay is applied
      
      % momentum at update t:= min(maxMomentum, (t + C)/(t + 2*C))
      % for example, t=0 => rho=1/2, t=C => rho=2/3, t=8C => rho=9/10
      C
      maxMomentum
      
      nUpdates = 0 
   end
   
   methods
      function obj = MomentumSchedule(lr0, maxMomentum, varargin)
         obj.params = {lr0, maxMomentum};
         obj.lr0 = lr0;
         obj.maxMomentum = maxMomentum;
         
         p = inputParser();
         p.addParamValue('lrDecay', []);
         p.addParamValue('lrBurnIn', 0);
         p.addParamValue('C', []);
         parse(p, varargin{:});
         
         obj.lrDecay = p.Results.lrDecay;
         obj.lrBurnIn = p.Results.lrBurnIn;
         obj.C = p.Results.C;
      end
      
      function params = update(obj)
         obj.nUpdates = obj.nUpdates + 1;
         
         % Decay learning rate if appropriate
         if ~isempty(obj.lrDecay) && obj.nUpdates > obj.lrBurnIn
            obj.params{1} = obj.lrDecay*obj.params{1};
         end
         
         % Set momentum
         if ~isempty(obj.C)
            obj.params{2} = min(obj.maxMomentum, ...
                                    (obj.nUpdates + obj.C)/(obj.nUpdates + 2*obj.C));
         end
         
         params = obj.params;
      end
      
      function reset(obj)
         obj.nUpdates = 0;
         obj.params = {obj.lr0, obj.maxMomentum};
      end
   end
   
end

