function [dy] = TCdyn(t,Y)


global gsyn
dy = zeros(size(Y));


N = 1;
for j = 1:N

    vth = Y(1 + 4*(j-1))*100;
    hth = Y(2 + 4*(j-1));
    rth = Y(3 + 4*(j-1))/100;
    sth = Y(4 + 4*(j-1));

    
    Cm = 1;
    gL = 0.05;
    EL = -70;
    gNa = 3;
    ENa = 50;
    gK = 5;
    EK = -90;
    gT = 5;
    ET = 0;
    hth_inf = 1/(1+exp((vth+41)/4));
    rth_inf = 1/(1+exp((vth+84)/4));
    ah = 0.128*exp(-(vth+46)/18);
    bh = 4/(1+exp(-(vth+23)/5));
    tauh = 1/(ah+bh);
    taur = 1*(28+exp(-(vth+25)/10.5));
    m_inf = 1/(1+exp(-(vth+37)/7));
    p_inf = 1/(1+exp(-(vth+60)/6.2));
Ib = 0.8;
%thalamic cell currents
    IL=gL*(vth-EL);
    INa=gNa*(m_inf.^3).*hth.*(vth-ENa);
    IK=gK*((0.75*(1-hth)).^4).*(vth-EK);
    IT=gT*(p_inf.^2).*rth.*(vth-ET);
    

    
    %synaptic
    alpha = 3;
Vt = -20;
sigmat = .8;
beta = .2;
Esyn = -100;
%%%


% gsyn = min(.1+ 1/10000*t,1.5);

    Isyn = 0;
%Differential Equations for cells thalamic
    vth_dot= 1/Cm*(-IL-INa-IK-IT+Ib-Isyn);   %- .01*(vth-sum(Y(1:4:4*N)));
    hth_dot=(hth_inf-hth)./tauh;
    rth_dot=(rth_inf-rth)./taur;
    sth_dot = alpha*(1-sth)*(1/(1+exp(-(vth-Vt)/sigmat)))-beta*sth;
    
    dy(1 + 4*(j-1)) = vth_dot/100;
    dy(2 + 4*(j-1)) = hth_dot;
    dy(3 + 4*(j-1)) = rth_dot*100;
    dy(4 + 4*(j-1)) = sth_dot;

    %myt = t;
    %XX = floor(myt/dt);
    %xrem = myt-XX*dt;
    
end

dy = dy(:);