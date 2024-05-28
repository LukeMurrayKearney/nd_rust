use crate::network_structure::NetworkStructure;
use crate::network_properties::{NetworkProperties, State};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use statrs::distribution::{Poisson,Exp};
use rand_distr::Distribution;
use rayon::prelude::*;


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
        step_tau_leap(network_structure, network_properties, &mut rng, scaling);
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
                        _ => step_tau_leap(network_structure, &mut network_properties, &mut rng, scaling)
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

fn step_tau_leap(network_structure: &NetworkStructure, network_properties: &mut NetworkProperties, rng: &mut ThreadRng, scaling: &str) {
    // save next states to update infection simultaneously 
    let mut next_states: Vec<State> = vec![State::Susceptible; network_structure.degrees.len()];
    // indices to update generation number
    let mut gen_idx: Vec<(usize, usize)> = Vec::new();
    // define random number generators fro each period
    let poisson_infectious_period = Poisson::new(network_properties.parameters[1]).unwrap();
    
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
                                next_states[link.1] = State::Infected(poisson_infectious_period.sample(rng).round() as usize);
                                // add to secondary cases for infected individual
                                network_properties.secondary_cases[i] += 1;
                                network_properties.disease_from[link.1] = i;
                                // update generation of target
                                gen_idx.push((link.1, network_properties.generation[i]));
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