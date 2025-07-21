using OrdinaryDiffEq
using FFTW
using Plots
using SparseArrays
using Distributed
import MAT

# useful funcs
heaviside(t) = 0.5 * (sign(t) + 1)
w(x, σ) = (1/(2σ))*exp(-abs(x)/σ)
wft(k, σ) = 1/(1+(2*π*k)^2*σ^2)
f(u) = (1+tanh(10.0.*u))/2 #heaviside(u) 
ψ(u,h,ŵ,N) = irfft(ŵ.*rfft(f.(u.-h)),N)

function D2x(R::Int64)
    #second order central difference operator
    A = spzeros(R, R);
    A[1:R + 1:end] .= -2.0
    A[R + 1:R + 1:end] .= 1.0
    A[2:R + 1:end] .= 1.0
    A[1,end] = 1.0
    A[end,1] = 1.0

    return A
end


function odefunc!(du, u, p, t)
    # model rhs
    N, ŵ, D, h0, γ, ∇2 = p
    ϕ = u[1:N]
    h = u[N+1:end]
    dϕ = du[1:N]
    dh = du[N+1:end]
    dϕ = -ϕ
    dϕ .+= ψ(ϕ, h, ŵ, N)
    dh = -(h.-h0) .+ D*∇2*h .+ γ*ϕ
    du .= [dϕ;dh]
end

function  getfrontspeed(sol, dx, N)
    # Find front speed from a simulation
    # select time mid way through simulation
    tpoints = length(sol.t)
    tmidpoint1 = 1
    uslice = sol[2N÷5:3N÷5,tmidpoint1]
    hslice = sol[2N÷5+N:3N÷5+N,tmidpoint1]
    frontidx1 = findmin(abs.(uslice.-hslice))[2]

    # repeat for later point
    tmidpoint2 = 2
    uslice2 = sol[2N÷5:3N÷5,tmidpoint2]
    hslice2 = sol[2N÷5+N:3N÷5+N,tmidpoint2]
    frontidx2 = findmin(abs.(uslice2.-hslice2))[2]
    return (frontidx2-frontidx1)*dx/(sol.t[tmidpoint2]-sol.t[tmidpoint1]), frontidx2
end


function step(N, h0)
    # step IC
    u0 = zeros(2N)
    u0[1:N÷2] .= 1.0
    u0[N÷2] = h0
    u0[N+1:end] .= h0
    u0[N+1:N+N÷2] .= 1.2h0
    u0[N+N÷2+2:end] .= 0.8h0 
    return u0
end

function bump(N, dx, width, amplitude,h0)
    # bump IC
    u0 = zeros(2N)
    u0[N+1:2N] .= h0
    m = floor(Int64,width/dx)
    u0[(N-m)÷2:(N+m)÷2] .= amplitude
    return u0
end

function front_speeds(N, xs, dx, σ, D, h0, γ, ∇2, freq, ŵ)
    # find front speeds as function of parameter
    u0 = step(N, h0)
    Ds = range(0.0,2.0,50)
    cs = zeros(50)
    for (i,D) ∈ enumerate(Ds)
        println("Running $i of 50")
        prob = ODEProblem(odefunc!, u0, [0.0,10.0], (N, ŵ, D, h0, γ, ∇2), saveat=[7.0,10.0])
        sol = solve(prob, Tsit5())
        cs[i], mid_index = getfrontspeed(sol, dx, N)
        println(cs[i])
    end
    mat_dict=Dict("D"=>collect(Ds),"c"=>cs)
    MAT.matwrite("D_c.mat", mat_dict)

end

function surface(N, xs, dx, σ, D, h0, γ, ∇2, freq, ŵ)
    # plot example surface
    prob = ODEProblem(odefunc!, u0, [0.0,10.0], (N, ŵ, D, h0, γ, ∇2), saveat=range(5.0,10.0,500))
    sol = solve(prob, Tsit5())
    mat_dict=Dict("ts"=>sol.t,"xs"=>collect(xs),"us"=>collect(sol[1:N,:]),"hs"=>collect(sol[N+1:2N,:]))
    MAT.matwrite("surface_data.mat", mat_dict)
end

function propogation_threshold(N, xs, dx, σ, D, h0, γ, ∇2, freq, ŵ)
    println("Starting simulation with γ=$γ")
    # compute the width threshold for a bump to propogate via a bisection method
    lower = 0.2 # lower threshold
    upper = 3 # upper threshold

    # bisection
    while upper-lower>3*dx
        l = (upper+lower)/2
        u0 = bump(N, dx, l, 1.0, h0)
        prob = ODEProblem(odefunc!, u0, [0.0,5.0], (N, ŵ, D, h0, γ, ∇2), saveat=[0.0,5.0])
        sol = solve(prob, Tsit5())
        propogates = sol[N÷5,end]>0.1
        if propogates
            upper = l
        else
            lower = l
        end
    end
    println("Simulation with γ=$γ complete")
    return (upper+lower)/2
end

function main()
    N = 25000 #gridpoints
    xs = range(-250,250,N) #domain
    dx = xs[2]-xs[1]
    σ = 1.0 
    D = 0.4
    h0 = 0.25
    γ = 0.4
    ∇2 = D2x(N)/dx^2
    freq = rfftfreq(N, 1/dx)
    ŵ = wft.(freq, σ)
    front_speeds(N, xs, dx, σ, D, h0, γ, ∇2, freq, ŵ)
    surface(N, xs, dx, σ, D, h0, γ, ∇2, freq, ŵ)
end

main()