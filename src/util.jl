function cheb_diff(N)
    x = [cos(pi*i/N) for i in 0:N]
    c = [2; ones(N-1); 2] .* (-1).^(0:N)
    D = zeros(N+1, N+1)
    for i in 0:N, j in 0:N
        if i != j
            D[i+1, j+1] = c[i+1]/c[j+1] / (x[i+1] - x[j+1])
        end
    end
    for i in 0:N
        D[i+1, i+1] = -sum(D[i+1, :])
    end
    return x, D
end