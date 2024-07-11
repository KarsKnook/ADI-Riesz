using PiecewiseOrthogonalPolynomials, NPZ

nmax = 10
pmax = 10

for n = 1:nmax
   for p = 1:pmax
      println("n = $(n), p = $(p)")

      r = range(-1, 1; length=n+1)
      C = ContinuousPolynomial{1}(r)  # hat functions and W_k = (1-x^2)P_k^(1,1)
      x = axes(C, 1)
      D = Derivative(x)
      M = C'*C  # mass matrix
      Delta = (D*C)'*(D*C)  # weak laplacian

      A = M[Block.(1:p+2),Block.(1:p+2)]  # upto polynomial degree p
      B = Delta[Block.(1:p+2),Block.(1:p+2)]  # upto polynomial degree p

      npzwrite("mass_$(n)_$(p).npz", A)
      npzwrite("weak_laplacian_$(n)_$(p).npz", B)
   end
end
