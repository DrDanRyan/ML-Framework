classdef ContractivePenalties < handle
   % A mixin that provides penalties from Rifai2011 "Higher Order Contractive
   % Autoencoders". Two penalties are available: the Frobenius norm of the
   % Jacobian of the mapping from input to output, and a finite-difference
   % approximation to the Hessian of the same map. It is assumed that the
   % nonlinearity in the layer is applied componentwise, y_i = F(z_i) (i.e. nothing like
   % softmax where y_i depends on the entire vector z).
   
   properties (Abstract)
      params
      gpuState
   end
   
   properties (Dependent = true, SetAccess = private)
      % boolean flag indicating if penalties are turned on; will be true if
      % JacobianPenalty is positive and false otherwise
      isContractionPenalty
   end 
   
   properties
      % a positive coefficient for Frobenius norm of Jacobian squared;
      % a term that shows up for Contractive Autoencoders. 
      JacobianPenalty 
      
      % a positive coefficient for finite difference approximation of scaled
      % approximation to Frobenius norm of Hessian squared; this applies the
      % form seen in Rifai2011 "Higher Order Contractive Autoencoders"
      HessianPenalty
      
      % number of noise perturbations per training example used to generate
      % Hessian approximation; only used if HessianPenalty and JacobianPenalty
      % are positive
      HessianBatchSize
      
      % standard deviation of random noise used to perturb inputs for Hessian
      % finite difference estimation
      HessianNoise
   end
   
   methods (Abstract)
      z = compute_z(x)
      [dydz, d2y_dz2] = compute_contractive_stats(obj, z)
   end
   
   methods
      function obj = ContractivePenalties(varargin)
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('JacobianPenalty', 0, @(x) x>=0);
         p.addParamValue('HessianPenalty', 0, @(x) x>=0);
         p.addParamValue('HessianBatchSize', 5);
         p.addParamValue('HessianNoise', .1);
         parse(p, varargin{:});
         
         obj.JacobianPenalty = p.Results.JacobianPenalty;
         obj.HessianPenalty = p.Results.HessianPenalty;
         obj.HessianBatchSize = p.Results.HessianBatchSize;
         obj.HessianNoise = p.Results.HessianNoise;
      end
      
      function flag = get.isContractionPenalty(obj)
         flag = obj.JacobianPenalty > 0;
      end
      
      function penalty = compute_contractive_penalty(obj, x, dydz, d2y_dz2)
         [L1, N] = size(x);
         H = obj.HessianBatchSize; % alias to shorten lines below
         W = obj.params{1};
         
         % Jacobian Penalty
         % d2y_dz2 empty implies second derivative is always zero for
         % this layer type.
         if isempty(d2y_dz2)
            penalty{1} = obj.JacobianPenalty*bsxfun(@times, W, ...
               mean(dydz.*dydz, 2));
            penalty{2} = 0;
         else % second derivative is needed for this layer type
            W_rowSquared = sum(W.*W, 2);
            dydz_d2ydz2_product = dydz.*d2y_dz2;
            penalty{2} = obj.JacobianPenalty*W_rowSquared.*...
                                             mean(dydz_d2ydz2_product, 2);
            
            x_outer_dydz_d2ydz2 = dydz_d2ydz2_product*x'/N;
            penalty{1} = obj.JacobianPenalty*(bsxfun(@times, W_rowSquared, ...
               x_outer_dydz_d2ydz2) + bsxfun(@times, W, mean(dydz.*dydz, 2)));
         end
         
         % Hessian Penalty
         if obj.HessianPenalty > 0
            xRepeated = repmat(x, 1, H); % L2 x (NH)
            xNoisy = xRepeated + obj.HessianNoise*obj.gpuState.randn([L1, N*H]);
            zNoisy = obj.compute_z(xNoisy);
            [dydzNoisy, d2y_dz2Noisy] = obj.compute_contractive_stats(zNoisy);
            clear zNoisy
            
            dydz_diff = repmat(dydz, 1, H) - dydzNoisy; % L2 x (NH)
            clear dydzNoisy
            
            if isempty(d2y_dz2)
               penalty{1} = penalty{1} + obj.HessianPenalty*bsxfun(@times, ...
                  W, mean(dydz_diff.*dydz_diff, 2));
            else
               d2y_dz2_Repeated = repmat(d2y_dz2, 1, H);
               d2y_dz2_diff = d2y_dz2_Repeated - d2y_dz2Noisy;
               
               penalty{2} = penalty{2} + obj.HessianPenalty*W_rowSquared.*...
                  mean(dydz_diff.*d2y_dz2_diff, 2);
               clear d2y_dz2_diff
               
               temp1 = ((dydz_diff.*d2y_dz2_Repeated)*xRepeated' - ...
                        (dydz_diff.*d2y_dz2Noisy)*xNoisy')/(N*H); % L2 x L1
               clear xRepeated xNoisy d2y_dz2Noisy d2y_dz2_Repeated
               
               penalty{1} = penalty{1} + obj.HessianPenalty*...
                  (bsxfun(@times, W_rowSquared, temp1) + ...
                   bsxfun(@times, W, mean(dydz_diff.*dydz_diff, 2)));
            end
         end
      end
      
   end   
end

