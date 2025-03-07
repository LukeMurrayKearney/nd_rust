// use crate::dpln::Parameters;
use crate::network_structure::NetworkStructure;
use crate::network_properties::{self, NetworkProperties, State};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
// use rand_distr::num_traits::{Pow, ToBytes};
use statrs::distribution::{Continuous, Exp, Geometric, Normal, Uniform};
use rand_distr::{Binomial, Distribution, WeightedIndex};
use rayon::prelude::*;
use statrs::statistics::Statistics;
use std::cmp;
use ndarray::{Array1, ArrayBase};

// pub struct ScaleParams {
//     pub a: Vec<f64>,
//     pub b: Vec<f64>,
//     pub c: Vec<f64>,
//     pub d: Vec<f64>,
//     pub e: Vec<f64>,
// }
pub struct ScaleParams {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
}

impl ScaleParams {
    // pub fn new(a: f64,b: f64, c: f64, d: f64, e: f64,) -> ScaleParams {
    //     ScaleParams {a:a, b:b, c:c, d:d, e:e}
    // }
    pub fn new(a: f64,b: f64, c: f64, d: f64, e: f64) -> ScaleParams {
        ScaleParams {a:a, b:b, c:c, d:d, e:e}
    }

    pub fn from_string(scaling: &str) -> ScaleParams {
        if scaling == "fit1" {
            ScaleParams::new(1.92943985e-01, 2.59700437e-01,4.55889377e04,9.99839680e-01,-4.55800575e04)
            // ScaleParams::new(
            //     vec![7.94597576e-02, 1.86501075e-02, 0.27331857, 2.71714397e-01, 1.50843120e-01, 0.22501698, 0.36498229,0.3402533,0.16861257],
            //     vec![1.86727109e-01, 1.57216691e-01, 0.26578535, 2.93698607e-01, 2.33321449e-01, 0.3025751, 0.32910101,0.32746589,0.23098462],
            //     vec![6.26714288e+04, 1.15274663e+05, 55.08602862, 1.01006951e+04, 2.61115659e+02, 123.57028328, 8.26235973,6.75177633,1.15683718], 
            //     vec![9.99749380e-01, 9.99844176e-01, 0.84128853, 9.99305162e-01, 9.71511671e-01, 0.93912153, 0.62576043,0.73053706,0.28046687],
            //     vec![-6.26598777e+04, -1.15266709e+05, -46.86310399, -1.00927696e+04, -2.51956809e+02, -115.13632869, 0.72628637,2.23407002,8.0201658])
        }
        else if scaling == "fit2" {
            ScaleParams::new(5.93853399e-02,1.81040353e-01,  1.08985503e+05,  9.99930465e-01, -1.08976101e+05)
            // ScaleParams::new(
            //     vec![6.87326840e-01, 7.38784056e-02, 5.77862944e-01, 2.13415641e-01, 3.29687844e-01, 3.85160330e-01, 3.24887201e-01,0.25216752,5.27970340e-02],
            //     vec![4.04847685e-01, 1.55839474e-01, 4.98656357e-01, 2.80893647e-01, 2.70297825e-01, 2.97786924e-01, 3.61132503e-01,0.35245878,1.71913746e-01],
            //     vec![9.50900768e+04, 1.94907963e+05, 4.96158310e+04, 5.52982066e+04, 2.69952333e+04, 2.34848788e+04, 7.52264235e+03,4.50828294,7.44934579e+03], 
            //     vec![9.99885451e-01, 9.99928224e-01, 9.99822099e-01, 9.99878834e-01, 9.99780538e-01, 9.99786125e-01, 9.99440989e-01,0.62107256,9.99746592e-01],
            //     vec![-9.50788242e+04, -1.94898582e+05, -4.96091268e+04, -5.52894549e+04, -2.69858020e+04, -2.34759408e+04, -7.51374782e+03,5.22464991,-7.43959415e+03])
        }
        else {
            ScaleParams::new(0., 0., 0., 0., 0.)
            // ScaleParams::new(vec![0.], vec![0.], vec![0.], vec![0.], vec![0.])
        }
    }
}


pub fn run_sellke(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, initially_infected: f64, scaling: &str) 
    -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>, Vec<i64>) {

    // seed an outbreak
    let n = network_structure.partitions.last().unwrap().to_owned();
    let mut rng = rand::thread_rng();
    network_properties.initialize_infection_sellke(network_structure, initially_infected, scaling);
    let scale_params = ScaleParams::from_string(scaling);
    // network_properties.initialize_infection_sellke_rand(initially_infected);
    // holding sir 
    let mut sir: Vec<Vec<usize>> = Vec::new();
    // let mut sir_ages: Vec<Vec<Vec<usize>>> = Vec::new();
    sir.push(network_properties.count_states());
    // sir_ages.push(network_properties.count_states_age(network_structure));

    // data structures for holding events
    let mut I_cur: Vec<usize> = network_properties.nodal_states
        .iter()
        .enumerate()
        .filter(|(_,&state)| state == State::Infected(0))
        .map(|(i,_)| i)
        .collect();
    let mut I_events: Vec<i64> = I_cur.iter().map(|&x| x as i64).collect();
    let mut R_events: Vec<i64> = vec![-1; I_events.len()];
    let mut t: Vec<f64> = vec![0.; I_events.len()];

    // defining outbreak parameters
    let beta = network_properties.parameters[0];
    let inv_gamma = network_properties.parameters[1];
    let mut ct = Array1::<f64>::zeros(n);
    // base infection pressure proportion ct on adjacency matrix 
    for &person in I_cur.iter() {
        update_ct(&mut ct, network_structure, true, person, scaling, &scale_params);
    }
    
    // define infection periods
    let exp_infectious = Exp::new(1./inv_gamma).unwrap();
    let I_periods: Vec<f64> = (0..n).map(|_| exp_infectious.sample(&mut rng)).collect();
    // println!("network = \n{:?}", network_structure);
    // println!("properties = \n{:?}", network_properties);
    // println!("average infectious period {:?}", I_periods.iter().sum::<f64>()/(I_periods.len() as f64));
    // println!("I_cur = \n{:?}", I_cur);

    // define thresholds
    let exp_thresh = Exp::new(1.).unwrap();
    let mut thresholds: Vec<f64> = (0..n).map(|_| exp_thresh.sample(&mut rng)).collect();
    // set thresholds of infected people to zero 
    for &i in I_cur.iter() {
        thresholds[i] = -1.;
    }

    // recovery times of everyone
    let mut recovery_times: Vec<(usize, f64)> = Vec::new();
    for &i in I_cur.iter() {
        recovery_times.push((i, I_periods[i]));
    }

    // define La_t and j,k 
    let mut tt = 0.;
    let mut La_t = Array1::<f64>::zeros(n);

    // start while loop 
    while I_cur.len() > 0 {
        // get the minimum recovery time
        // println!("\nlength of R = {:?}", recovery_times.len());
        let (min_index_vec, min_index_node, min_r) = recovery_times
            .iter()
            .enumerate()
            .min_by(|(_,a),(_,b)| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, a)| (i,a.0,a.1))
            .unwrap();

        // time before first recovery
        let dtprop = min_r - tt;
        // change in lam in that time
        let lambda = dtprop * beta * ct.clone();
        let mut Laprop = &La_t + &lambda;

        // println!("min index node = {:?} \nminR = {:?}\n",min_index_node, min_r);
        // println!("dtprop = {:?} \nlambda = \n{:?}\n",dtprop,lambda);
        // println!("Laprop = \n{:?}\n",Laprop);


        
        // if only recoveries left, S=0
        if sir.last().unwrap()[0] == 0 {
            // println!("none left\n");
            // println!("Recovery\nsum(LA_t) = {:?}\nsum(ct) = {:?}\n",La_t.iter().filter(|&&x| x>=0.).sum::<f64>(), ct.iter().sum::<f64>());
            recovery_times.remove(min_index_vec);
            tt = min_r;
            t.push(min_r);
            // update SIR and event vecs
            I_events.push(-1);
            R_events.push(min_index_node as i64);
            I_cur = I_cur.iter().filter(|&&x| x != min_index_node).map(|&x| x).collect::<Vec<usize>>();
            update_sir(&mut sir, false);
            // update_sir_ages(&mut sir_ages, false, network_structure.ages[min_index_node]);
            La_t = Laprop.clone();
            // update ct 
            update_ct(&mut ct, &network_structure, false, min_index_node, scaling,&scale_params);
        }
        else {
            // we may have multiple infections before a recovery,
            // to get correct increase in FOI we need to do these in the right order 
            let mut waiting_infections: Vec<usize> = Vec::new();
            for (i, &threshold) in thresholds.iter().enumerate().filter(|(_,&x)| x>=0.) {
                // println!("threshold = {threshold}");
                // println!("i = {i}\n");
                // infection event 
                if threshold < Laprop[i] {
                    waiting_infections.push(i);
                }
            }
            // if no infections pending before recovery
            if waiting_infections.len() == 0 {
                // println!("Recovery\nsum(LA_t) = {:?}\nsum(ct) = {:?}\n",La_t.iter().filter(|&&x| x>=0.).sum::<f64>(), ct.iter().sum::<f64>());
                // do recovery
                recovery_times.remove(min_index_vec);
                tt = min_r.clone();
                t.push(min_r.clone());
                // update SIR and event vecs
                I_events.push(-1);
                R_events.push(min_index_node as i64);
                I_cur = I_cur.iter().filter(|&&x| x != min_index_node).map(|&x| x).collect::<Vec<usize>>();
                update_sir(&mut sir, false);
                La_t = Laprop.clone();
                update_ct(&mut ct, &network_structure, false, min_index_node, scaling, &scale_params);
            }
            // do infection
            else {
                // println!("Infection\nsum(LA_t) = {:?}\nsum(ct) = {:?}\n",La_t.iter().filter(|&&x| x>=0.).sum::<f64>(), ct.iter().sum::<f64>());
                // we need to find which threshold would break first, not trivial because of network structure
                let first_infection = waiting_infections
                    .iter()
                    .max_by(|&&a,&&b| (Laprop[a]/thresholds[a]).partial_cmp(&(Laprop[b]/thresholds[b])).unwrap())
                    .unwrap()
                    .to_owned();
                // time of first infection
                let ratio = (thresholds[first_infection] - La_t[first_infection])/(Laprop[first_infection] - La_t[first_infection]);
                tt = tt + ratio*dtprop;
                // println!("first infection = {:?}\nratio = {:?}\n", first_infection, ratio);
                t.push(tt.clone());
                // set a new La to be used at the start of next iteration
                // let ratio = thresholds[first_infection]/Laprop[first_infection];
                thresholds[first_infection] = -1.;
                La_t = &La_t + &lambda*ratio;
                // add their recovery time
                recovery_times.push((first_infection, I_periods[first_infection] + tt));
                
                // add info on secondary cases generation and infection from 
                // to choose secondary case pick randomly with probability based on each persons FOI on i
                let contacts = network_structure.adjacency_matrix[first_infection]
                    .iter()
                    .map(|(_, x)| x.to_owned())
                    .collect::<Vec<usize>>();
                let impacts: Vec<f64> = contacts
                    .iter()
                    .map(|&j|{
                        // this is wrong, we cannot be infected by someone who isnt infected oops
                        // else if R_events.contains(&(j as i64)) {
                        //     // calculate how long the neighbour was infected for.. 
                        //     let time_infec = t[R_events.iter().position(|&x| x == (j as i64)).unwrap()] - t[I_events.iter().position(|&x| x == (j as i64)).unwrap()];
                            
                        //     return single_FOI((network_structure.degrees[first_infection], network_structure.degrees[j]), scaling,&scale_params) * time_infec
                        // }
                        // if neighbour infected
                        if I_events.contains(&(j as i64)) && !R_events.contains(&(j as i64)){
                            let time_infec = tt - t[I_events.iter().position(|&x| x == (j as i64)).unwrap()];
                            return single_FOI((network_structure.degrees[first_infection], network_structure.degrees[j]), scaling, &scale_params)
                        }
                        else {
                            return 0.;
                        }
                    })
                    .collect();
                // println!("contacts: {:?}\nimpacts: {:?}\nI_events: {:?}\nR_events: {:?}",contacts, impacts, I_events, R_events);
                let dist = WeightedIndex::new(&impacts).unwrap();
                let index_case = contacts[dist.sample(&mut rng)];
                network_properties.disease_from[first_infection] = index_case as i64;
                network_properties.generation[first_infection] = network_properties.generation[index_case] + 1;
                network_properties.secondary_cases[index_case] += 1;
                // println!("index case: {:?}\nfirst infection: {:?}", index_case, first_infection);
                // println!("index case links: {:?}\nfirst infection links: {:?}", network_structure.adjacency_matrix[index_case].iter().map(|x| x.1).collect::<Vec<usize>>(), network_structure.adjacency_matrix[first_infection].iter().map(|x| x.1).collect::<Vec<usize>>());
                // println!("generations: {:?}, {:?}", network_properties.generation[index_case], network_properties.generation[first_infection]);
                // println!("\n\nstep\n\n");

                // update SIR and event vecs
                I_events.push(first_infection as i64);
                R_events.push(-1);
                I_cur.push(first_infection);
                update_sir(&mut sir, true);
                // update_sir_ages(&mut sir_ages, true, network_structure.ages[first_infection]);
                update_ct(&mut ct, &network_structure, true, first_infection, scaling,&scale_params);
            }
        }
        // println!("t = \n{:?}",t);
        // println!("\n\nstep\n\n");
    }
    // println!("I_cur = {:?}\nI_events = {:?}\nR_events = {:?}\n", I_cur, I_events, R_events);
    // println!("t = {:?}",t);
    // println!("{:?}", sir.last().unwrap());
    (t, I_events, R_events, sir, network_properties.secondary_cases.clone(), network_properties.generation.clone(), network_properties.disease_from.clone())
}

fn single_FOI(degrees: (usize,usize), scaling: &str, scale_params: &ScaleParams) -> f64 {
    match scaling {
        "fit1" => {
            let k = cmp::max(degrees.0, degrees.1);
            // change in c for this link scaled
            1. * (scale_fit(&scale_params, k as f64) / scale_fit(&scale_params, 1.))
        }
        "fit2" => {
            let k = cmp::max(degrees.0, degrees.1);
            // change in c for this link scaled
            1. * (scale_fit(&scale_params, k as f64) / scale_fit(&scale_params, 1.))
        }
        _ => {
            1.
        }
    }
}

fn update_ct(ct: &mut Array1<f64>, network: &NetworkStructure, infection: bool, i: usize, scaling: &str, scale_params: &ScaleParams) {
    
    for link in network.adjacency_matrix[i].iter() {
        // we want to decide which side and if we are scaling 
        match scaling {
            "fit1" => {
                let k = cmp::max(network.degrees[link.0], network.degrees[link.1]);
                // change in c for this link scaled
                let dc = 1. * (scale_fit(&scale_params, k as f64) / scale_fit(&scale_params, 1.));
                // infection event
                if infection == true {
                    ct[link.1] += dc;
                }
                else {
                    ct[link.1] -= dc;
                }
            }
            "fit2" => {
                let k = cmp::max(network.degrees[link.0], network.degrees[link.1]);
                // change in c for this link scaled
                let dc = 1. * (scale_fit(&scale_params, k as f64) / scale_fit(&scale_params, 1.));
                // infection event
                if infection == true {
                    ct[link.1] += dc;
                }
                // recovery event
                else {
                    ct[link.1] -= dc;
                }
            }
            _ => {
                //infection event
                if infection == true {
                    ct[link.1] += 1.;
                }
                // recovery event
                else {
                    ct[link.1] -= 1.;
                }
            }
        }
    }
}

fn update_sir(sir: &mut Vec<Vec<usize>>, infection: bool) {
    if infection == true {
        let mut tmp = sir.last().unwrap().to_owned();
        tmp[0] -= 1; tmp[1] += 1;
        sir.push(tmp);
    }
    else {
        let mut tmp = sir.last().unwrap().to_owned();
        tmp[1] -= 1; tmp[2] += 1;
        sir.push(tmp);
    }
}

fn update_sir_ages(sir_ages: &mut Vec<Vec<Vec<usize>>>, infection: bool, age: usize) {
    if infection == true {
        let mut tmp = sir_ages.last().unwrap().to_owned();
        tmp[age][0] -= 1;
        tmp[age][1] += 1;
        sir_ages.push(tmp);
    }
    else {
        let mut tmp = sir_ages.last().unwrap().to_owned();
        tmp[age][1] -= 1;
        tmp[age][2] += 1;
        sir_ages.push(tmp);
    }
}

pub fn fit_to_hosp_data(data: Vec<f64>, days: Vec<usize>, tau_0: f64, proportion_hosp: f64, iters: usize, dist_type: &str, n: usize, partitions: &Vec<usize>, contact_matrix: &Vec<Vec<f64>>, network_params: &Vec<Vec<f64>>, outbreak_params: &Vec<f64>, prior_param: f64, scaling: &str) 
    -> (Vec<f64>, f64) {

    // define priors and random number generator
    let exp_prior = Exp::new(prior_param).unwrap();
    let uniform = Uniform::new(0., 1.).unwrap();
    // let rng = rand::thread_rng();
    // define vector of taus 
    let mut taus: Vec<f64> = vec![0.; iters+1];
    taus[0] = tau_0;
    
    // start point of adaptive mcmc
    // let n0 = 100;
    // define variance in random pulls
    // let (mut mu, mut sigma, mut ll) = (0., 0.002, 0.);
    
    let (sigma, mut ll, mut num_accept) = (0.003, 0., 0);
    let mut rng: ThreadRng = rand::thread_rng();
    // iterate over mcmc chain length
    for i in 1..(iters+1) {
        if i % 100 == 0 {
            println!("{i}");
        }
        // generate a new proposal for tau using optimal scaling result
        let normal = if sigma > 1e-8 {
            Normal::new(taus[i-1], (2.38f64).powi(2)*sigma/3.)
        }
        else if sigma < 1e-8{
            Normal::new(taus[i-1], -(2.38f64).powi(2)*sigma/3.)
        }
        else{
            Normal::new(taus[i-1], 1e-8)
        };
        let proposal = normal.unwrap().sample(&mut rng);
        // new log likelihood
        let ll_new = log_likelihood_incidence(&data, &days, n, partitions, network_params, outbreak_params, contact_matrix, dist_type, proposal, proportion_hosp, scaling);
        // calculate the log acceptance ratio, including priors
        let l_acc = ll_new - ll + exp_prior.ln_pdf(proposal) - exp_prior.ln_pdf(taus[i-1]);
        ll = ll_new;
        // generate random number for acceptance criteria
        if uniform.sample(&mut rng).ln() < l_acc {
            // accept proposal
            taus[i] = proposal;
            num_accept += 1;
        }
        else {
            taus[i] = taus[i-1];
        }
        /////////////// not going to use adaptive part
        // // println!("tau: {}, \n\nproposal: {proposal}, \n\nll_new: {ll_new}, \n\nl_acc: {l_acc}, \n\nll: {ll}", taus[i-1]);
        // // the adaptive part, changing variance of pulls 
        // if i == n0 {
        //     mu = taus.iter().take(i).mean();
        //     sigma = taus.iter().take(i).variance() + 1e-6;
        // }
        // else if i > n0 {
        //     let i_float = i as f64;
        //     let mu_old = mu;
        //     mu = (i_float*mu + taus[i])/(i_float + 1.);
        //     sigma = sigma*(i_float-1.)/i_float + taus[i].powi(2) + i_float*mu_old.powi(2) - (i_float + 1.)*mu.powi(2) + 1e-6/i_float;
        // }


    }
    (taus, (num_accept as f64)/(iters as f64))
}

fn log_likelihood_incidence(data: &Vec<f64>, days: &Vec<usize>, n: usize, partitions: &Vec<usize>, network_params: &Vec<Vec<f64>>, outbreak_params: &Vec<f64>, contact_matrix: &Vec<Vec<f64>>, dist_type: &str, tau: f64, proportion_hosp: f64, scaling: &str) -> f64 {

    let mut ll = 0.;
    if tau >= 0. && tau < 1. {
        let parameters = outbreak_params.iter().enumerate().map(|(i, &x)| {
            if i == 0 {tau} else {x}
        }).collect::<Vec<f64>>();
        // simulate outbreak using tau value ten times 
        let lls: Vec<f64> = vec![0; 10].par_iter()
            .map(|_|{
                let mut ll_tmp = 0.;
                let (mut new_infections, mut hospitalisations) = (vec![0; days.last().unwrap().to_owned()], vec![0; days.last().unwrap().to_owned()]);
                // create network
                let network: NetworkStructure = if dist_type == "sbm" {
                    NetworkStructure::new_sbm_from_vars(n, partitions, contact_matrix)
                }
                else {
                    NetworkStructure::new_mult_from_input(n, partitions, dist_type, network_params, contact_matrix)
                };
                //  initialise infection and parameterize outbreak
                let mut properties = NetworkProperties::new(&network, &parameters);
                properties.initialize_infection_degree(&network, 1./(network.degrees.len() as f64), 5., scaling);
                let mut rng = rand::thread_rng();
                let (mut takeoff, mut num_restarts) = (false, 0);
                while takeoff == false {
                    //simulate outbreak
                    for i in 0..days.last().unwrap().to_owned() {
                        new_infections[i] = step_tau_leap(&network, &mut properties, &mut rng, scaling);

                        // cannot use poisson because data is not integer valued
                        // let pois = Poisson::new(x);
                        // ll += pois.unwrap().pmf(data[i].round() as u64).ln();
                        let binomial = Binomial::new(new_infections[i] as u64, proportion_hosp);
                        hospitalisations[i] = binomial.unwrap().sample(&mut rng);
                    }
                    // find number of hospitalisations a week and calculate likelihood
                    for (i, &sample) in days.iter().enumerate() {
                        let mut tmp_hosp = 0;
                        for day in 0..sample {
                            tmp_hosp += hospitalisations[day];
                        }
                        let normal = if tmp_hosp == 0 {
                            Normal::new(tmp_hosp as f64, 1.).unwrap()
                        }
                        else {
                            Normal::new(tmp_hosp as f64, tmp_hosp as f64).unwrap()
                        };

                        ll_tmp += normal.pdf(data[i]).ln();
                    }
                    // if outbreak doesn't take off  retry a max of 5 times
                    if new_infections.iter().skip(days.last().unwrap()/2).sum::<usize>() == 0 {
                        num_restarts += 1;
                        if num_restarts < 5 {
                            continue;
                        }
                    }
                    takeoff = true;
                }
                ll_tmp
            })
            .collect::<Vec<f64>>();
        ll += lls.iter().mean();
    }
    else {
        ll = - f64::INFINITY;
    }
    ll
}

pub fn quick_run(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, maxtime: usize, initially_infected: f64, scaling: &str) ->
    (usize, usize, f64) {

    // infect a proportion of the population
    network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[1], scaling);
    let mut rng: ThreadRng = rand::thread_rng();
    let mut seir_results: Vec<Vec<usize>> = Vec::new();
    seir_results.push(network_properties.count_states());
    let mut new_cases = Vec::new();

    //start simulation
    for _ in 0..maxtime {
        // step a day
        new_cases.push(step_tau_leap(network_structure, network_properties, &mut rng, scaling));
        // collect summary days summer stats
        seir_results.push(network_properties.count_states());
        // check if there is infetion in the population
        if seir_results.last().unwrap()[1] == 0 {
            break;
        }
    }
    // calc r0
    let gen4: Vec<usize> = network_properties.generation
        .iter()
        .enumerate()
        .filter(|(_, &x)| x == 4)
        .map(|(i, _)| network_properties.secondary_cases[i] )
        .collect();
    let gen23: Vec<usize> = network_properties.generation
        .iter()
        .enumerate()
        .filter(|(_, &x)| x == 2 || x == 3)
        .map(|(i, _)| network_properties.secondary_cases[i] )
        .collect();
    let r0 = if gen4.len() < 2 {-1.} else {(gen23.iter().sum::<usize>() as f64) / (gen23.len() as f64)};
    (seir_results.last().unwrap()[2], seir_results.iter().map(|row| row[1]).max().unwrap(), r0)
}

pub fn run_tau_leap(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, maxtime: usize, initially_infected: f64, scaling: &str) -> 
    (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>, Vec<i64>, Vec<usize>)  {

    // infect a proportion of the population
    network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[1], scaling);
    let mut rng: ThreadRng = rand::thread_rng();
    // create vector to store results
    let mut individual_results: Vec<Vec<usize>> = Vec::new(); 
    // summary results
    let mut seir_results: Vec<Vec<usize>> = Vec::new();
    let mut new_cases = Vec::new();

    //start simulation
    for _ in 0..maxtime+1 {
        // collect summary days summer stats
        seir_results.push(network_properties.count_states());
        // append all individual states to results at the end of the day
        individual_results.push(
            network_properties.nodal_states
                .iter()
                .map(|x| {
                    match x {
                        State::Susceptible => 0,
                        State::Infected(_) => 1,
                        _ => 2
                    }
                })
                .collect()
        );

        // step a day
        new_cases.push(step_tau_leap(network_structure, network_properties, &mut rng, scaling));
        
        // check if there is infetion in the population
        if seir_results.last().unwrap()[1] == 0 {
            break;
        }
    }
    (individual_results, seir_results, network_properties.generation.clone(), network_properties.secondary_cases.clone(), network_properties.disease_from.clone(), new_cases)
}


pub fn abc_r0(network_structure: &NetworkStructure, properties: &mut NetworkProperties, initially_infected: f64, target_r0: f64, iters: usize, n: usize, cavity: bool, scaling: &str) -> f64 {
       
    // randomly sample a vector of Exply distributed values for prior
    let mut rng = rand::thread_rng();
    let exp = Exp::new(10.).unwrap(); // lambda = 5, gives P(x<1) > 0.99
    let samples: Vec<f64> = (0..iters).map(|_| exp.sample(&mut rng)).collect();

    let r0s = r0_from_taus(network_structure, properties, initially_infected, n, &samples, cavity, scaling);
    

    // find the best few and find the best of these with more samples
    let mut indexed_vals: Vec<_> = r0s.into_iter().map(|r0| (target_r0 - r0).abs()).enumerate().collect();
    indexed_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());


    let curr_best = indexed_vals.into_iter().take(5).map(|(i,_)| {
        samples[i]
        }).collect();
    
    let r0s = r0_from_taus(network_structure, properties, initially_infected, 2*n, &curr_best, cavity, scaling);
    
    //return minimum
    let r0_differences: Vec<f64> = r0s.iter().map(|&r0| (r0 - target_r0).abs()).collect();
    let min_index = r0_differences.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    
    curr_best[min_index.unwrap().0]

}

fn r0_from_taus(network_structure: &NetworkStructure, properties: &mut NetworkProperties, initially_infected: f64, n: usize, samples: &Vec<f64>, cavity: bool, scaling: &str) -> Vec<f64> {
    // loop through samples in parallel
    let r0s: Vec<f64> = samples.par_iter()
        .map(|&tau| {
            let mut rng = rand::thread_rng();
            if tau > 1. {
                return f64::INFINITY;
            }
            let mut count = 0; // this is to ensure an infection can take off
            let mut results: Vec<f64> = Vec::new();
            while results.len() < n { // number of samples to reduce variance
                count += 1;
                let mut network_properties = properties.clone();
                network_properties.parameters[0] = tau;
                // infect a proportion of the population
                network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[1], scaling);
                let mut secondary_cases: Vec<usize> = Vec::new();
                //start simulation
                for day in 0..10_000 {
                    // step a day
                    match cavity {
                        true => step_cavity(network_structure, &mut network_properties, &mut rng, day + 1),
                        _ => _ = step_tau_leap(network_structure, &mut network_properties, &mut rng, scaling)
                    }

                    // check if there are any generation 3 people still infected
                    let gen3: Vec<usize> = network_properties.generation
                        .iter()
                        .enumerate()
                        .filter(|(_, &x)| x == 2 || x == 3)
                        .map(|(i, _)| i)
                        .collect();

                    let gen23: Vec<usize> = network_properties.generation
                        .iter()
                        .enumerate()
                        .filter(|(_, &x)| x == 2 || x == 3)
                        .map(|(i, _)| i)
                        .collect();

                    if !gen3.is_empty() {
                        // check if still infected
                        if gen23.iter().filter(|&i| {match network_properties.nodal_states[*i] {State::Infected(_) => true, _ => false}}).map(|_| true).collect::<Vec<bool>>().len() == 0 { 
                            secondary_cases = gen23
                                .iter()
                                .map(|&i| network_properties.secondary_cases[i])
                                .collect();
                            break;
                        }
                    }
                    let sir = network_properties.count_states();
                    // check if epidemic has died
                    if sir[1] == 0 {
                        break;
                    }
                }
                // calculate r0
                if !secondary_cases.is_empty() {
                    let r0 = (secondary_cases.iter().sum::<usize>() as f64) / (secondary_cases.len() as f64);
                    results.push(r0);
                }
                if count >= 5*n {
                    results.push(f64::INFINITY);
                }
            }    
            results.iter().sum::<f64>() / (results.len() as f64)
        }).collect();
    r0s
}

fn step_tau_leap(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, rng: &mut ThreadRng, scaling: &str) -> usize {
    // save next states to update infection simultaneously 
    let mut next_states: Vec<State> = vec![State::Susceptible; network_structure.degrees.len()];
    let mut new_infections = 0;
    // indices to update generation number
    let mut gen_idx: Vec<(usize, usize)> = Vec::new();
    // define random number generators fro each period
    // let poisson_infectious_period = Poisson::new(network_properties.parameters[1]).unwrap();
    let geom_infectious_period = Geometric::new(1./network_properties.parameters[1]).unwrap();
    // define parameters of fitted scaling
    let scale_params = if scaling == "fit1" {
        ScaleParams::new(1.92943985e-01, 2.59700437e-01,4.55889377e04,9.99839680e-01,-4.55800575e04)
        // ScaleParams::new(
        //     vec![7.94597576e-02, 1.86501075e-02, 0.27331857, 2.71714397e-01, 1.50843120e-01, 0.22501698, 0.36498229,0.3402533,0.16861257],
        //     vec![1.86727109e-01, 1.57216691e-01, 0.26578535, 2.93698607e-01, 2.33321449e-01, 0.3025751, 0.32910101,0.32746589,0.23098462],
        //     vec![6.26714288e+04, 1.15274663e+05, 55.08602862, 1.01006951e+04, 2.61115659e+02, 123.57028328, 8.26235973,6.75177633,1.15683718], 
        //     vec![9.99749380e-01, 9.99844176e-01, 0.84128853, 9.99305162e-01, 9.71511671e-01, 0.93912153, 0.62576043,0.73053706,0.28046687],
        //     vec![-6.26598777e+04, -1.15266709e+05, -46.86310399, -1.00927696e+04, -2.51956809e+02, -115.13632869, 0.72628637,2.23407002,8.0201658])
    }
    else if scaling == "fit2" {
        ScaleParams::new(5.93853399e-02,1.81040353e-01,  1.08985503e+05,  9.99930465e-01, -1.08976101e+05)
        // ScaleParams::new(
        //     vec![6.87326840e-01, 7.38784056e-02, 5.77862944e-01, 2.13415641e-01, 3.29687844e-01, 3.85160330e-01, 3.24887201e-01,0.25216752,5.27970340e-02],
        //     vec![4.04847685e-01, 1.55839474e-01, 4.98656357e-01, 2.80893647e-01, 2.70297825e-01, 2.97786924e-01, 3.61132503e-01,0.35245878,1.71913746e-01],
        //     vec![9.50900768e+04, 1.94907963e+05, 4.96158310e+04, 5.52982066e+04, 2.69952333e+04, 2.34848788e+04, 7.52264235e+03,4.50828294,7.44934579e+03], 
        //     vec![9.99885451e-01, 9.99928224e-01, 9.99822099e-01, 9.99878834e-01, 9.99780538e-01, 9.99786125e-01, 9.99440989e-01,0.62107256,9.99746592e-01],
        //     vec![-9.50788242e+04, -1.94898582e+05, -4.96091268e+04, -5.52894549e+04, -2.69858020e+04, -2.34759408e+04, -7.51374782e+03,5.22464991,-7.43959415e+03])
    }
    else {
        ScaleParams::new(0., 0., 0., 0., 0.)
        // ScaleParams::new(vec![0.], vec![0.], vec![0.], vec![0.], vec![0.])
    };
    
    for (i, state) in network_properties.nodal_states.iter().enumerate() {

        match *state {
            State::Susceptible => (),
            State::Infected(days) => {
                //infection countdown
                if days <= 1 {
                    // // SEIRS
                    // next_states[i] = State::Recovered(poisson_recovered_period.sample(rng).round() as usize);
                    // SEIR
                    next_states[i] = State::Recovered;
                }
                else {
                    next_states[i] = State::Infected(days - 1);
                }
                // find connections to infected individuals
                for link in network_structure.adjacency_matrix[i].iter() {
                    match network_properties.nodal_states[link.1] {
                        State::Susceptible => {
                            // check if infection is passed
                            let infection_prob = match scaling {
                                "log" => {
                                    let k = network_structure.degrees[i];
                                    if k > 1 {
                                        network_properties.parameters[0] / (k as f64).ln()
                                    }
                                    else {
                                        network_properties.parameters[0]
                                    }
                                }
                                "sqrt" => network_properties.parameters[0] / (network_structure.degrees[i] as f64).sqrt(),
                                "linear" => network_properties.parameters[0] / (network_structure.degrees[i] as f64),
                                // duration = A*(k^2)*exp(-Bk) + C/(k^D) + E/k 
                                // max at 1, when k=1 we want f(k) = 1
                                "fit1" => {
                                    // use the k max on either side of link
                                    let (_, k) = if network_structure.degrees[i] > network_structure.degrees[link.1] {
                                        (i, network_structure.degrees[i] as f64)
                                    }
                                    else if network_structure.degrees[i] < network_structure.degrees[link.1] {
                                        (link.1, network_structure.degrees[link.1] as f64)
                                    }
                                    else {
                                        if rng.gen_bool(0.5) {(i, network_structure.degrees[i] as f64)} 
                                        else {(link.1, network_structure.degrees[link.1] as f64)}
                                    };
                                    network_properties.parameters[0] * (scale_fit(&scale_params, k) / scale_fit(&scale_params, 1.))
                                },
                                "fit2" => {
                                    let (_, k) = if network_structure.degrees[i] > network_structure.degrees[link.1] {
                                        (i, network_structure.degrees[i] as f64)
                                    }
                                    else if network_structure.degrees[i] < network_structure.degrees[link.1] {
                                        (link.1, network_structure.degrees[link.1] as f64)
                                    }
                                    else {
                                        if rng.gen_bool(0.5) {(i, network_structure.degrees[i] as f64)} 
                                        else {(link.1, network_structure.degrees[link.1] as f64)}
                                    };
                                    network_properties.parameters[0] * (scale_fit(&scale_params, k / scale_fit(&scale_params, 1.)))
                                },
                                _ => network_properties.parameters[0]
                            };
                            if rng.gen::<f64>() < infection_prob {
                                // make infected at next step
                                next_states[link.1] = State::Infected(geom_infectious_period.sample(rng).round() as usize);
                                // add to secondary cases for infected individual
                                network_properties.secondary_cases[i] += 1;
                                network_properties.disease_from[link.1] = i as i64;
                                // update generation of target
                                gen_idx.push((link.1, network_properties.generation[i]));
                                new_infections += 1;
                            }
                        },
                        _ => ()
                    }
                }
            },
            State::Recovered => {
                // // if you want simulation to be SEIRS
                // if days <= 0 {
                //     next_states[i] = State::Susceptible;
                // }
                // else {
                //     next_states[i] = State::Recovered(days - 1);
                // }
                next_states[i] = State::Recovered;
            }
        }
    }
    for (i, gen) in gen_idx.iter() {
        network_properties.generation[*i] = gen + 1
    }
    network_properties.nodal_states = next_states;

    new_infections
}

pub fn scale_fit(params: &ScaleParams, k: f64) -> f64 {
    params.a*(-params.b*k).exp()*k.powi(2) + params.c/k.powf(params.d) + params.e/k
}

// pub fn run_cavity(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, maxtime: usize, initially_infected:f64) -> 
//     (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>, Vec<usize>) { 

//     // infect a proportion of the population
//     let mut rng: ThreadRng = rand::thread_rng();
//     network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[2]);
//     let mut individual_results: Vec<Vec<usize>> = Vec::new(); 
//     individual_results.push(
//         network_properties.nodal_states
//             .iter()
//             .map(|x| {
//                 match x {
//                     State::Susceptible => 0,
//                     State::Infected(_) => 1,
//                     _ => 2
//                 }
//             })
//             .collect()
//     );
//     // summary results
//     let mut seir_results: Vec<Vec<usize>> = Vec::new();
//     seir_results.push(network_properties.count_states());

//     //start simulation
//     for generation in 0..maxtime {
//         // step a generation
//         step_cavity(network_structure, network_properties, &mut rng, generation + 1);

//         // append all individual states to results at the end of the day
//         individual_results.push(
//             network_properties.nodal_states
//                 .iter()
//                 .map(|x| {
//                     match x {
//                         State::Susceptible => 0,
//                         State::Infected(_) => 1,
//                         _ => 2
//                     }
//                 })
//                 .collect()
//         );

//         // collect summary days summer stats
//         seir_results.push(network_properties.count_states());
//         // check if there is infetion in the population
//         if seir_results.last().unwrap()[1] == 0 {
//             break;
//         }
//     }
//     (individual_results, seir_results, network_properties.generation.clone(), network_properties.secondary_cases.clone(), network_properties.disease_from.clone())
// }

fn step_cavity(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, rng: &mut ThreadRng, generation: usize)  {

    // save next states to update infection simultaneously 
    let mut next_states: Vec<State> = vec![State::Susceptible; network_structure.degrees.len()];

    let mut indices = network_properties.nodal_states.iter().enumerate().collect::<Vec<(usize, &State)>>();
    indices.shuffle(rng);
    for (i, state) in indices.iter() {
        match *state {
            State::Susceptible => (),
            State::Infected(_) => {
                // no infection countdown
                next_states[*i] = State::Recovered;
                // find connections to infected individuals
                for link in network_structure.adjacency_matrix[*i].iter() {
                    match network_properties.nodal_states[link.1] {
                        State::Susceptible => {
                            // check if infection is passed
                            if rng.gen::<f64>() < network_properties.parameters[0] {
                                // make infected at next step
                                next_states[link.1] = State::Infected(1);
                                // add to secondary cases for infected individual
                                network_properties.secondary_cases[*i] += 1;
                                network_properties.disease_from[link.1] = *i as i64;
                                network_properties.generation[link.1] = generation + 1;
                            }
                        },
                        _ => ()
                    }
                }
            },
            State::Recovered => {
                next_states[*i] = State::Recovered;
            }
        }
    }
    network_properties.nodal_states = next_states;
}