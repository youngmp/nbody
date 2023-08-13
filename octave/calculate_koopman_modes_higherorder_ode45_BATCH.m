global perorb myfn dt kappa numberderivs g1 g2 g3 g4 g5 g6 g7 g8 z0 z1 z2 z3 z4 z5 z6 z7 z8 i0 i1 i2 i3 i4 i5 i6 i7 i8 kappaother H2fn gfirst gsecond myorb PRCORIRC
global baseinp

T=38.7;
dt = .004;
options = odeset('abstol',1e-10,'reltol',1e-10);

global rfn zfn
load symbolicrs
load symbolicrs_zs

%myfn = @tcdyn;
myfn = @tcdyn_s2;
global tshift

%% get LC 
dt = .01;
[t,y] = ode45(myfn,[0:dt:1500],[0 0 0 0],options);

ww = length(y)-1;
while(  (y(ww,1)+.4)*(y(ww+1,1)+.4) > 0   || y(ww,1)>y(ww+1,1))
    ww = ww-1;
end
ww1 = ww;
ww = ww-10;
while(  (y(ww,1)+.4)*(y(ww+1,1)+.4)   > 0   || y(ww,1)>y(ww+1,1)  )
    ww = ww-1;
end
ww2 = ww;


%baseinptake = -.005:.0002:.014;
% baseinptake = baseinptake - .0001;
%for qq = 1:length(baseinptake)
%    baseinp = baseinptake(qq);

T=dt*(ww1-ww2);
options = odeset('abstol',1e-13,'reltol',1e-13);
init = y(ww1,:);
init = [init T];

% [t,Y] = ode45('circmodel',[0 1000],init(1:3),options);

Y = init;
Y = Y(:);
dy = ones(5,1);its = 0;
while norm(dy)>1e-12

    if its>0
    [t,Yorig] = ode45(myfn,[0 Y(5)],Y(1:4),options);
    [aa bb] = max(Yorig(:,1));
    Y(1:4) = Yorig(bb,:);
        end
        its = its+1;
        
    clear J
    tspan = [0 Y(5)];
    %%%Calculate Jacobian
    eps = .0011;
    cplus = Y(1:4) + [eps 0 0 0]';
    cminus = Y(1:4) - [eps 0 0 0]';
    [t,Yplus] = ode45(myfn,tspan,cplus,options);
    [t,Yminus] = ode45(myfn,tspan,cminus,options);
    mapplus = Yplus(length(Yplus),:);
    
    mapminus = Yminus(length(Yminus),:);
    J(:,1) = [(mapplus-mapminus)/(2*eps)]';
    
    cplus = Y(1:4) + [0 eps 0 0]';
    cminus = Y(1:4) - [0 eps 0 0]';
    [t,Yplus] = ode45(myfn,tspan,cplus,options);
    [t,Yminus] = ode45(myfn,tspan,cminus,options);
    mapplus = Yplus(length(Yplus),:);
    mapminus = Yminus(length(Yminus),:);
    J(:,2) = [(mapplus-mapminus)/(2*eps)]';
    
    cplus = Y(1:4) + [0 0 eps 0]';
    cminus = Y(1:4) - [0 0 eps 0]';
    [t,Yplus] = ode45(myfn,tspan,cplus,options);
    [t,Yminus] = ode45(myfn,tspan,cminus,options);
    mapplus = Yplus(length(Yplus),:);
    mapminus = Yminus(length(Yminus),:);
    J(:,3) = [(mapplus-mapminus)/(2*eps)]';
    
    
    cplus = Y(1:4) + [0 0  0 eps]';
    cminus = Y(1:4) - [0 0   0 eps]';
    [t,Yplus] = ode45(myfn,tspan,cplus,options);
    [t,Yminus] = ode45(myfn,tspan,cminus,options);
    mapplus = Yplus(length(Yplus),:);
    mapminus = Yminus(length(Yminus),:);
    J(:,4) = [(mapplus-mapminus)/(2*eps)]';
    
    J = J-eye(4);
    
    %%%%Calculate Tdiff
    epstime = 0.0001;
    [t,Yplus] = ode45(myfn,[0 (Y(5)+epstime)],Y(1:4),options);
    [t,Yminus] = ode45(myfn,[0 (Y(5)-epstime)],Y(1:4),options);
    mapplus = Yplus(length(Yplus),:);
    mapminus = Yminus(length(Yminus),:);
    J(:,5) = [(mapplus-mapminus)/(2*epstime)]';
    J(5,:) = [myfn(0,Y(1:4))' 0];
    
    [t,Yorig] = ode45(myfn,[0 Y(5)],Y(1:4),options);
    Yafter = Yorig(length(Yorig),:)';
    b = [Y(1:4)-Yafter;0];
    
    dy = J\b;
    Y = Y+dy;
    %[dy Y]
    norm(dy)

end
 


T = Y(5);
mytime = linspace(0,Y(5),10000);
dt = mytime(2)-mytime(1);
init = Y(1:4);
[t,Y] = ode45(myfn,mytime,init,options);
perorb = Y;

%shg

[argval,agmax] = max(Y);

init = perorb(1,:);

%% get monodromy matrix

%options = odeset('abstol',1e-10,'reltol',1e-10);
[t,Y] = ode45(@monodromyfn,mytime,eye(length(init)),options);
buf = Y(length(Y),:).';
buf = reshape(buf,length(init),length(init));

monodromy_mat = buf;
[V,D] = eig(buf);
% load vsave;D = D.^.5;
VV = V
DD = D;
abs(diag(D))



[aa,bb] = sort(diag(D),'descend');
fa = bb(2);

if sum(VV(:,fa))<0
    VV(:,fa) = -VV(:,fa);
end
V = VV;

lambda = (D(fa,fa));
kappa = log(lambda)/T
omega = 2*pi/T;

disp("g1 next")

%% Get g1
IRCinit = VV(:,fa)/2;
Zd = IRCinit;
myorb = perorb';
d = 1;

options = odeset('abstol',1e-10,'reltol',1e-10);


%getrc(1,[1,1,1,1]')
for mm = 1:1
    [t,Zd] = ode45(@getrc,mytime,Zd,options);
    Zd = Zd.';
end

Zd(:,1)./Zd(:,length(Zd));
g1 = Zd;
clear Zd;

clear J;
g2 = zeros(size(g1));
g3 = zeros(size(g1));
g4 = zeros(size(g1));
g5 = zeros(size(g1));
g6 = zeros(size(g1));
g7 = zeros(size(g1));
g8 = zeros(size(g1));

disp("g2 next")
%%get g2
init = 0*init(:);
myorb = perorb';
numberderivs = 2;
%calculatehigherorders(1,[1,1,1,1]')
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@calculatehigherorders,mytime,Zd_u,options);
    Zd_u = Zd_u.';
    

    if mm<5
        
        eps = .1;
        for p = 1:length(init)
            
            pert = zeros(length(init),1);
            pert(p) = eps;
            Zd_p = init + pert;
            d = 1;
            
            [t,Zd_p] = ode45(@calculatehigherorders,mytime,Zd_p,options);    
            Zd_p = Zd_p.';
            
            J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
            [numberderivs mm p];
        end

        mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
        dx = -(J-eye(length(init)))\mydiff;
        norm(dx)
        dx;
        mydiff;
        [Zd_u(:,1) Zd_u(:,end)];
        init = init + dx;
    end
end

g2 = Zd_u;


disp("g3 next")
%{
%% get g3
init = 0*init(:);
myorb = perorb';
numberderivs = 3;
calculatehigherorders(1,[1,1,1,1]');

for mm = 1:10
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@calculatehigherorders,mytime,Zd_u,options);
    Zd_u = Zd_u.';

if mm<10
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@calculatehigherorders,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p];
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff];
norm(dx)
init = init + dx;
end

end

g3 = Zd_u;



%%get g4
init = 0*init(:);
myorb = perorb';
numberderivs = 4;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:4
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@calculatehigherorders,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<4
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@calculatehigherorders,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end

end
plot(Zd_u(1,:));hold on;getframe
g4 = Zd_u;

%%get g5
init = 0*init(:);
myorb = perorb';
numberderivs = 5;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:4
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@calculatehigherorders,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<4
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@calculatehigherorders,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end

end
plot(Zd_u(1,:));hold on;getframe
g5 = Zd_u;


%%get g6
init = 0*init(:);
myorb = perorb';
numberderivs = 6;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:4
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@calculatehigherorders,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<4
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@calculatehigherorders,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end

end
plot(Zd_u(1,:));hold on;getframe
g6 = Zd_u;


subplot(4,1,1)
plot(g1(1,:))
ylabel('g1_V')

subplot(4,1,2)
plot(g2(1,:))
ylabel('g2_V')

subplot(4,1,3)
plot(g3(1,:))
ylabel('g3_V')

subplot(4,1,4)
plot(g4(1,:))
ylabel('g4_V')


%}
g1 = fliplr(g1);
g2 = fliplr(g2);
g3 = fliplr(g3);
%g4 = fliplr(g4);
%g5 = fliplr(g5);
%g6 = fliplr(g6);
%g7 = fliplr(g7);
%g8 = fliplr(g8);

z0 = zeros(size(perorb.'));
z1 = zeros(size(z0));
z2 = zeros(size(z1));
z3 = zeros(size(z1));
z4 = zeros(size(z1));
z5 = zeros(size(z1));
z6 = zeros(size(z1));
z7 = zeros(size(z1));
z8 = zeros(size(z1));

i0 = zeros(size(perorb.'));
i1 = zeros(size(i0));
i2 = zeros(size(i1));
i3 = zeros(size(i1));
i4 = zeros(size(i1));
i5 = zeros(size(i1));
i6 = zeros(size(i1));
i7 = zeros(size(i1));
i8 = zeros(size(i1));


%% get z0
disp("z0 next")

%%needs to be normalized%%%
WW = inv(VV/2);
%WW = inv(VV);
PRCORIRC = 1;
buf = diag(DD);
[a1,a2] = min(abs(buf-1));
IRCinit = WW(a2,:);
Zd = IRCinit;
myorb = flipud(perorb);numberderivs = 0;
options = odeset('abstol',1e-10,'reltol',1e-10);
d = 1;
for mm = 1:1
    [t,Zd] = ode45(@getprcirc,mytime,Zd,options);
    Zd = Zd.';
end
%need to normalize
buf = myfn(0,perorb(1,:));
%Zd = omega * Zd/(buf.'*Zd(:,1));
Zd = 1 * Zd/(buf.'*Zd(:,1));

Zd(:,1)./Zd(:,length(Zd));
z0 = Zd;
clear Zd;


clear J
disp("z1 next")

%% get z1
%%get higher orders
init = zeros(length(init),1);
numberderivs = 1;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:10
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
    if mm<10
        eps = .1;
        for p = 1:length(init)
        
            pert = zeros(length(init),1);
            pert(p) = eps;
            Zd_p = init + pert;
            d = 1;
           
            [t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
            Zd_p = Zd_p.';
            
            J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
            [numberderivs mm p];
        end
        mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
        dx = -(J-eye(length(init)))\mydiff;
        [dx mydiff];
        norm(dx)
        init = init + dx;
    end
end
z1 = Zd_u;

disp("z2 next")

%% get z2
%%get higher orders
init = zeros(length(init),1);
numberderivs = 2;
%options = odeset('abstol',1e-10,'reltol',1e-10);
for mm = 1:10
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
    if mm<10
        eps = .1;
        for p = 1:length(init)
        
            pert = zeros(length(init),1);
            pert(p) = eps;
            Zd_p = init + pert;
            d = 1;
           
            [t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
            Zd_p = Zd_p.';
            
            J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
            [numberderivs mm p];
        end
        mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
        dx = -(J-eye(length(init)))\mydiff;
        [dx mydiff];
        norm(dx)
        init = init + dx;
    end
end

z2 = Zd_u;


%{
disp("z3 next")

%% get z3
%%get higher orders
init = zeros(length(init),1);
numberderivs = 3;
%options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:10
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<10
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p];
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff];
norm(dx)
init = init + dx;
end
end

z3 = Zd_u;


%%get higher orders
init = zeros(length(init),1);
numberderivs = 4;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
z4 = Zd_u;




%%get higher orders
init = zeros(length(init),1);
numberderivs = 5;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
z5 = Zd_u;


%%get higher orders
init = zeros(length(init),1);
numberderivs = 6;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
z6 = Zd_u;
%}


%% get i0
disp("i0 next")


%%now get ircs%%
WW = inv(VV/2);
PRCORIRC = 2;
IRCinit = WW(fa,:);
Zd = IRCinit;
myorb = flipud(perorb);numberderivs = 0;
options = odeset('abstol',1e-10,'reltol',1e-10);
d = 1;
for mm = 1:1
    [t,Zd] = ode45(@getprcirc,mytime,Zd,options);
    Zd = Zd.';
end
Zd(:,1);
Zd = Zd/(Zd(:,1).'*g1(:,1));
%Zd(:,1)./Zd(:,length(Zd))
i0 = Zd;
clear Zd;

disp("i1 next")
%%get higher orders

%% get i1
init = zeros(length(init),1);
numberderivs = 1;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:10
    mm;
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
    Zd_u(:,end);
    
    if mm<10
        eps = 1;
        for p = 1:length(init)
            pert = zeros(length(init),1);
            pert(p) = eps;
            Zd_p = init + pert;
            d = 1;
           
            [t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
            Zd_p = Zd_p.';
            J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
            [numberderivs mm p];
        end
        mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
        
        mydiff(length(mydiff)+1) = 0;
        Jdiff = -(J-eye(length(init)));
        Jdiff(length(Jdiff)+1,:) = [1 0 0 0];
        dx = Jdiff\mydiff;
        norm(dx)
        [dx mydiff(1:4)];
        init = init + dx;
    
        if mm==10
            actual = myfn(0,perorb(1,:)).'*Zd_u(:,1);
            expect = i0(:,1).'*(kappa*eye(length(init)) - numericaljacobian(myfn,perorb(1,:)))*g1(:,1);
            canchg = z0(:,1);
            amtchg = myfn(0,perorb(1,:)).'*canchg;
            mymult = (actual-expect)/amtchg;
            init = Zd_u(:,1) - mymult*canchg;
        end
    end
    
    Zd_u = init;
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);

end

Zd_u = Zd_u.';
%plot(Zd_u(1,:));
i1 = Zd_u;


%% get  i2
disp("i2 next")

%%get higher orders
init = zeros(length(init),1);
numberderivs = 2;

for mm = 1:10
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<10
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p];
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff];
norm(dx)
init = init + dx;
end
end
i2 = Zd_u;

%{

%% get i3
disp("i3 next")

%%get higher orders
init = zeros(length(init),1);
numberderivs = 3;

for mm = 1:20
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<20
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p];
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff];
norm(dx)
init = init + dx;
end
end

i3 = Zd_u;



%%get higher orders
init = zeros(length(init),1);
numberderivs = 4;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
i4 = Zd_u;






%%get higher orders
init = zeros(length(init),1);
numberderivs = 5;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
i5 = Zd_u;




%%get higher orders
init = zeros(length(init),1);
numberderivs = 6;
options = odeset('abstol',1e-8,'reltol',1e-8);
for mm = 1:5
    d = 1;
    Zd_u = init;
    
    [t,Zd_u] = ode45(@getprcirc,mytime,Zd_u,options);
    Zd_u = Zd_u.';
if mm<5
eps = .1;
for p = 1:length(init)

    pert = zeros(length(init),1);
    pert(p) = eps;
    Zd_p = init + pert;
    d = 1;
   
[t,Zd_p] = ode45(@getprcirc,mytime,Zd_p,options);    
Zd_p = Zd_p.';
    
    J(:,p) = ( Zd_p(:,length(Zd_p)) - Zd_u(:,length(Zd_u)) ).'/eps;
    [numberderivs mm p]
end
mydiff = Zd_u(:,length(Zd_u))-Zd_u(:,1);
dx = -(J-eye(length(init)))\mydiff;
[dx mydiff]
init = init + dx;
end
end
plot(Zd_u(1,:));hold on;getframe
i6 = Zd_u;

%}

%%%%flip everything for forward time
z0 = fliplr(z0);
z1 = fliplr(z1);
z2 = fliplr(z2);
z3 = fliplr(z3);
%z4 = fliplr(z4);
%z5 = fliplr(z5);
%z6 = fliplr(z6);
%z7 = fliplr(z7);
%z8 = fliplr(z8);

%%%%flip everything for forward time
i0 = fliplr(i0);
i1 = fliplr(i1);
i2 = fliplr(i2);
i3 = fliplr(i3);
%i4 = fliplr(i4);
%i5 = fliplr(i5);
%i6 = fliplr(i6);
%i7 = fliplr(i7);
%i8 = fliplr(i8);


%%%%flip everything for forward time
g1 = fliplr(g1);
g2 = fliplr(g2);
g3 = fliplr(g3);
%g4 = fliplr(g4);
%g5 = fliplr(g5);
%g6 = fliplr(g6);



save('reducedfunctions_s2','kappa','i0','i1','i2','g1','g2','z0','z1','z2','T','omega','perorb','monodromy_mat')
return

