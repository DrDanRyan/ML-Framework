function W = sparse_init(M, N, nConnections, gpuState)
% Returns an M x N matrix with largest singular value approx 1.1 and
% nConnections nonzero entries in each row

   W = gpuState.zeros(M, N);
   for i = 1:M
      W(i, randperm(N, nConnections)) = gpuState.randn(1, nConnections);
   end
   singVals = svd(W);
   W = W*1.1/singVals(1);  
end

