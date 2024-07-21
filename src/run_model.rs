// use crate::dpln::Parameters;
use crate::network_structure::NetworkStructure;
use crate::network_properties::{NetworkProperties, State};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
// use rand_distr::num_traits::{Pow, ToBytes};
use statrs::distribution::{Continuous, Discrete, Exp, Geometric, Normal, Poisson, Uniform};
use rand_distr::{uniform, Binomial, Distribution};
use rayon::prelude::*;
use statrs::statistics::Statistics;

pub fn fit_to_hosp_data(data: Vec<f64>, days: Vec<usize>, tau_0: f64, proportion_hosp: f64, iters: usize, dist_type: &str, n: usize, partitions: &Vec<usize>, contact_matrix: &Vec<Vec<f64>>, network_params: &Vec<Vec<f64>>, outbreak_params: &Vec<f64>, prior_param: f64) 
    -> Vec<f64> {

    // define priors and random number generator
    let exp_prior = Exp::new(prior_param).unwrap();
    let uniform = Uniform::new(0., 1.).unwrap();
    let rng = rand::thread_rng();
    // define vector of taus 
    let mut taus: Vec<f64> = vec![0.; iters+1];
    taus[0] = tau_0;
    // start point of adaptive mcmc
    let n0 = 100;
    // define variance in random pulls
    let (mut mu, mut sigma, mut ll) = (0.1, 0.1, 0.);
    let mut rng: ThreadRng = rand::thread_rng();
    // iterate over mcmc chain length
    for i in 1..(iters+1) {
        if i % 1_000 == 0 {
            println!("{i}");
        }
        // generate a new proposal for tau using optimal scaling result
        let normal = Normal::new(taus[i-1], (2.38f64).powi(2)*sigma/3.);
        let proposal = normal.unwrap().sample(&mut rng);
        // new log likelihood
        let ll_new = log_likelihood_incidence(&data, &days, n, partitions, network_params, outbreak_params, contact_matrix, dist_type, proposal, proportion_hosp);
        // calculate the log acceptance ratio, including priors
        let l_acc = ll_new - ll + exp_prior.ln_pdf(proposal) - exp_prior.ln_pdf(taus[i-1]);
        ll = ll_new;
        // generate random number for acceptance criteria
        if uniform.sample(&mut rng).ln() < l_acc {
            // accept proposal
            taus[i] = proposal;
        }
        else {
            taus[i] = taus[i-1];
        }
        // the adaptive part, changing variance of pulls 
        if i == n0 {
            mu = taus.iter().take(i).mean();
            sigma = taus.iter().take(i).variance() + 1e-6;
        }
        else if i > n0 {
            let i_float = i as f64;
            let mu_old = mu;
            mu = (i_float*mu + taus[i])/(i_float + 1.);
            sigma = sigma*(i_float-1.)/i_float + taus[i].powi(2) + i_float*mu_old.powi(2) - (i_float + 1.)*mu.powi(2) + 1e-6/i_float;
        }
    }
    taus
}

fn log_likelihood_incidence(data: &Vec<f64>, days: &Vec<usize>, n: usize, partitions: &Vec<usize>, network_params: &Vec<Vec<f64>>, outbreak_params: &Vec<f64>, contact_matrix: &Vec<Vec<f64>>, dist_type: &str, tau: f64, proportion_hosp: f64) -> f64 {

    let mut ll = 0.;
    if tau <= 0. {
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
                properties.initialize_infection_degree(&network, 1./(network.degrees.len() as f64), 5.);
                let mut rng = rand::thread_rng();
                let (mut takeoff, mut num_restarts) = (false, 0);
                while takeoff == false {
                    //simulate outbreak
                    for i in 0..days.last().unwrap().to_owned() {
                        new_infections[i] = step_tau_leap(&network, &mut properties, &mut rng, "");

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

    }
    else {
        ll = - f64::INFINITY;
    }
    ll
}


pub fn run_tau_leap(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, maxtime: usize, initially_infected: f64, scaling: &str) -> 
    (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>, Vec<usize>)  {

    // infect a proportion of the population
    network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[1]);
    let mut rng: ThreadRng = rand::thread_rng();
    // create vector to store results
    let mut individual_results: Vec<Vec<usize>> = Vec::new(); 
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
    // summary results
    let mut seir_results: Vec<Vec<usize>> = Vec::new();
    seir_results.push(network_properties.count_states());

    //start simulation
    for _ in 0..maxtime {
        // step a day
        _ = step_tau_leap(network_structure, network_properties, &mut rng, scaling);
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
        // collect summary days summer stats
        seir_results.push(network_properties.count_states());
        // check if there is infetion in the population
        if seir_results.last().unwrap()[1] == 0 {
            break;
        }
    }
    (individual_results, seir_results, network_properties.generation.clone(), network_properties.secondary_cases.clone(), network_properties.disease_from.clone())
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
                network_properties.initialize_infection_degree(network_structure, initially_infected, network_properties.parameters[1]);
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
    
    for (i, state) in network_properties.nodal_states.iter().enumerate() {

        match *state {
            State::Susceptible => (),
            State::Infected(days) => {
                //infection countdown
                if days <= 0 {
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
                                "log" => network_properties.parameters[0] / (network_structure.degrees[i] as f64).ln(),
                                "sqrt" => network_properties.parameters[0] / (network_structure.degrees[i] as f64).sqrt(),
                                "linear" => network_properties.parameters[0] / (network_structure.degrees[i] as f64),
                                _ => network_properties.parameters[0]
                            };
                            if rng.gen::<f64>() < infection_prob {
                                // make infected at next step
                                next_states[link.1] = State::Infected(geom_infectious_period.sample(rng).round() as usize);
                                // add to secondary cases for infected individual
                                network_properties.secondary_cases[i] += 1;
                                network_properties.disease_from[link.1] = i;
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
                                network_properties.disease_from[link.1] = *i;
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