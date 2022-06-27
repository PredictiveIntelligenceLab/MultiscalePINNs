%% Gray-Scott equations in 2D
% Nick Trefethen, April 2016

%%
% (Chebfun Example pde/GrayScott.m)
% [Tags: #Gray-Scott, #spin2]

%% 1. Rolls
% The Gray-Scott equations are a pair of coupled reaction-diffusion
% equations that lead to interesting patterns [1,2,3].
% Let us look at two examples in 2D.

%%
% The equations are
% $$ u_t = \varepsilon_1\Delta u + b(1-u) - uv^2, \quad
% v_t = \varepsilon_2\Delta v - dv + uv^2, $$
% where $\Delta$ is the Laplacian and $\varepsilon_1,
% \varepsilon_2,b,d$ are parameters.
% To begin with we choose these values.
ep1 = 0.00001; ep2 = 0.000005;
b = 0.04; d = 0.1;
%%
% We now solve up to $t=3500$ with `spin2` and plot the $v$ variable.
% What beautiful, random-seeming "rolls" (or
% "fingerprints") appear!  
nn = 400;
steps = 500;
dt = 0.5;

dom = [-1 1 -1 1]; x = chebfun('x',dom(1:2)); tspan = linspace(0,5000, steps+1);
S = spinop2(dom,tspan);
S.lin = @(u,v) [ep1*lap(u); ep2*lap(v)];
S.nonlin = @(u,v) [b*(1-u)-u.*v.^2;-d*v+u.*v.^2];
S.init = chebfun2v(@(x,y) 1-exp(-80*((x+.05).^2+(y+.02).^2)), ...
                   @(x,y) exp(-80*((x-.05).^2+(y-.02).^2)),dom);
tic, u = spin2(S, nn, dt,'plot','off');

% plot(u{1, 4}), view(0,90), axis equal, axis off
time_in_seconds = toc

N = 200;
[X,Y] = meshgrid(linspace(-1,1, N), linspace(-1,1, N));

usol = zeros(N, N, steps+1);
for i = 1:steps+1
    usol(:,:,i) = u{1, i}(X,Y);
end

vsol = zeros(N,N, steps+1);
for i = 1:steps+1
    vsol(:,:,i) = u{2, i}(X,Y);
end

% save('sol.mat', 'b', 'd', 'ep1', 'ep2', 'tspan', 'usol', 'vsol', 'X', 'Y')

