data{
	int<lower=0> N; # number of patients
	int<lower=0> K; # number of covariates
	int<lower=0> L; # total number of datapoints

	int<lower=0> ls[N];
	real<lower=0> ts[L];
	real vs[L];

	vector[K] xs[N];
	real<lower=0.0,upper=1.0> ss[N];

	int<lower=0> N_test; # number of patients
#	int<lower=0> K_test; # number of covariates
#	int<lower=0> L_test; # total number of datapoints

#	int<lower=0> ls_test[N_test];
#	real<lower=0> ts_test[L_test];
#	real vs_test[L_test];

	vector[K] xs_test[N_test];
	real<lower=0.0,upper=1.0> ss_test[N_test];

	real<lower=0,upper=1> pop_a;	
	real<lower=0,upper=1> pop_b;	
	real<lower=0> pop_c;	

	real<lower=0> s_a;
	real<lower=0> s_b;
	real<lower=0> s_c;

#	real<lower=0> l_a;
#	real<lower=0> l_b;
#	real<lower=0> l_c;
	real<lower=0> l_m;

	# if fixing phi parameters, uncomment these, remove l_a, l_b, l_c, l_m
	real<lower=0,upper=1> phi_a;
	real<lower=0,upper=1> phi_b;
	real<lower=0,upper=1> phi_c;

}
parameters{
	vector[K] B_a;
	vector[K] B_b;
	vector[K] B_c;
	
	#real<lower=0,upper=1> phi_a;
	#real<lower=0,upper=1> phi_b;
	#real<lower=0,upper=1> phi_c;
	real<lower=0,upper=1> phi_m;
	
	real<lower=0,upper=1> as[N];	
	real<lower=0,upper=1> bs[N];
	real<lower=0> cs[N];

	real<lower=0,upper=1> as_test[N_test];	
	real<lower=0,upper=1> bs_test[N_test];
	real<lower=0> cs_test[N_test];

	real<lower=0,upper=1> p;
	real<lower=0,upper=1> theta;
}
transformed parameters{

	real phi_a_s;
	real phi_b_s;
	real phi_c_s;
	real phi_m_s;

	real<lower=0,upper=1> m_as[N];
	real<lower=0,upper=1> m_bs[N];
	real<lower=0> m_cs[N];

	real<lower=0,upper=1> m_as_test[N_test];
	real<lower=0,upper=1> m_bs_test[N_test];
	real<lower=0> m_cs_test[N_test];

	for(i in 1:N){
	      m_as[i] <- inv_logit(logit(pop_a) + dot_product(xs[i], B_a));		      
	      m_bs[i] <- inv_logit(logit(pop_b) + dot_product(xs[i], B_b));
	      m_cs[i] <- exp(log(pop_c) + dot_product(xs[i], B_c));
	}

	for(i in 1:N_test){
	      m_as_test[i] <- inv_logit(logit(pop_a) + dot_product(xs_test[i], B_a));		      
	      m_bs_test[i] <- inv_logit(logit(pop_b) + dot_product(xs_test[i], B_b));
	      m_cs_test[i] <- exp(log(pop_c) + dot_product(xs_test[i], B_c));
	}
	
	phi_a_s <- 1.0 / phi_a - 1;
	phi_b_s <- 1.0 / phi_b - 1;
	phi_c_s <- 1.0 / phi_c;
	phi_m_s <- 1.0 / phi_m - 1;
	


}
model{
	int c;
	real v;
	real eps;
	real low_eps;
	real high_eps;
	int indicator;
	int val_indicator;

	c <- 1;

	eps <- 0.001;
	low_eps <- eps;
	high_eps <- 1.0 - eps;
	

	B_a ~ normal(0, s_a);
	B_b ~ normal(0, s_b);
	B_c ~ normal(0, s_c);

	#phi_a ~ exponential(l_a);
	#phi_b ~ exponential(l_b);
	#phi_c ~ exponential(l_c);
	phi_m ~ exponential(l_m);

	for(i in 1:N){
	      as[i] ~ beta(1.0 + (phi_a_s * m_as[i]), 1.0 + (phi_a_s * (1.0 - m_as[i])));
	      bs[i] ~ beta(1.0 + (phi_b_s * m_bs[i]), 1.0 + (phi_b_s * (1.0 - m_bs[i])));
	      cs[i] ~ gamma(phi_c_s, (phi_c_s - 1.0) / m_cs[i]);
	}

	for(i in 1:N_test){
	      as_test[i] ~ beta(1.0 + (phi_a_s * m_as_test[i]), 1.0 + (phi_a_s * (1.0 - m_as_test[i])));
	      bs_test[i] ~ beta(1.0 + (phi_b_s * m_bs_test[i]), 1.0 + (phi_b_s * (1.0 - m_bs_test[i])));
	      cs_test[i] ~ gamma(phi_c_s, (phi_c_s - 1.0) / m_cs_test[i]);
	}

	p ~ uniform(0,1);
	theta ~ uniform(0,1);

	for(i in 1:N){
	      for(j in 1:ls[i]){
	      	    if (vs[c] < low_eps ){
		         val_indicator <- 0;
		         val_indicator ~ bernoulli(p);
			 indicator <- 1;
		    }
		    else if(vs[c] > high_eps){
		         val_indicator <- 1;
			 val_indicator ~ bernoulli(p);
			 indicator <- 1;
		    }
		    else{
			 v <- ss[i] * (1.0 - as[i] - (bs[i] * (1.0 - as[i]) * exp(-1.0 * ts[c] / cs[i])));
		    	 vs[c] ~ beta(1.0 + (phi_m_s * v), 1.0 + (phi_m_s * (1.0 - v)));
			 indicator <- 0;
		    }
		    indicator ~ bernoulli(theta);
		    c <- c + 1;
	      }
	}

}