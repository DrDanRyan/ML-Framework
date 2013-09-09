function W = dense_init(M, N, scale, gpuState)
   W = scale*gpuState.randn(M, N);
end

