use crate::network_structure::NetworkStructure;
use crate::run_model::{scale_fit, ScaleParams};
use rand::distributions::WeightedIndex;
use statrs::distribution::Poisson;
use rand_distr::{Distribution, Geometric};

#[derive(Clone, Debug)]
pub struct NetworkProperties {
    pub nodal_states: Vec<State>,
    pub parameters: Vec<f64>,
    pub secondary_cases: Vec<usize>,
    pub generation: Vec<usize>,
    pub disease_from: Vec<usize>,
}

#[derive(Clone,Debug)]
pub enum State {
    Susceptible,
    Infected(usize),
    Recovered
}

impl NetworkProperties {

    pub fn new(network: &NetworkStructure, params: &Vec<f64>) -> NetworkProperties {
        NetworkProperties {
            nodal_states: vec![State::Susceptible; network.degrees.len()],
            parameters: params.clone(),
            secondary_cases: vec![0; network.degrees.len()],
            generation: vec![0; network.degrees.len()],
            disease_from: (0..network.degrees.len()).collect::<Vec<usize>>(),
        }
    }

    pub fn initialize_infection_degree(&mut self, network: &NetworkStructure, proportion_of_population: f64, inv_gamma: f64, scaling: &str) {

        let number_of_infecteds: usize = match proportion_of_population as usize {
            0..=1 => {
                ((self.nodal_states.len() as f64) * proportion_of_population) as usize
            },
            _ => {
                println!("The proportion infected must be between 0 and 1");
                0
            }
        };

        // define infectious period sampler
        // let poisson_infectious_period = Poisson::new(inv_gamma).unwrap();
        let geom_infectious_period = Geometric::new(1./inv_gamma).unwrap();

        // define random number generator
        let mut rng = rand::thread_rng();

        
        //we want a weighted sampling of the population
        let probabilities: Vec<f64> = match scaling {
            "log" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).ln()).collect()
            },
            "sqrt" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).sqrt()).collect()
            },
            "linear" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
            },
            // duration = A*(k^2)*exp(-Bk) + C/(k^D) + E/k 
            // max at 1, when k=1 we want f(k) = 1
            "fit1" => {
                let scale_params = ScaleParams::new(1.92943985e-01, 2.59700437e-01,4.55889377e04,9.99839680e-01,-4.55800575e04);
                network.degrees.iter().map(|&degrees| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            "fit2" => {
                let scale_params = ScaleParams::new(5.93853399e-02,1.81040353e-01,  1.08985503e+05,  9.99930465e-01, -1.08976101e+05);
                network.degrees.iter().map(|&degrees| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            _ => network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
        };
        
        // weighted index of each individual
        println!("{:?}", probabilities.iter().enumerate().filter(|(_, &x)| !(x>0.)).map(|(i, _)| network.degrees[i]).collect::<Vec<usize>>());
        let dist = WeightedIndex::new(probabilities).unwrap();
        let selected: Vec<usize> = (0..number_of_infecteds).map(|_| dist.sample(&mut rng)).collect();

        // infect selected individuals
        for &i in selected.iter() {
            self.nodal_states[i] = State::Infected(geom_infectious_period.sample(&mut rng) as usize);
            // self.nodal_states[i] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
            self.generation[i] = 1;
        }
    }

    // pub fn initialize_infection_random(&mut self, proportion_of_population: f64, inv_gamma: f64) {

    //     let number_of_infecteds: usize = match proportion_of_population as usize {
    //         0..=1 => {
    //             ((self.nodal_states.len() as f64) * proportion_of_population) as usize
    //         },
    //         _ => {
    //             println!("The proportion infected must be between 0 and 1");
    //             0
    //         }
    //     };

    //     // define infectious period sampler
    //     let poisson_infectious_period = Poisson::new(inv_gamma).unwrap();
    //     // define random number generator
    //     let mut rng = rand::thread_rng();
    //     // shuffle indices and choose
    //     let mut indices: Vec<usize> = (0..self.nodal_states.len()).collect();
    //     indices.shuffle(&mut rng);
    //     for i in 0..number_of_infecteds {
    //         self.nodal_states[indices[i]] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
    //         self.generation[indices[i]] = 1;
    //     }
    // }

    pub fn count_states(&self) -> Vec<usize> {
        let mut result: Vec<usize> = vec![0; 3];
        for state in self.nodal_states.iter() {
            match state {
                State::Susceptible => result[0] += 1,
                State::Infected(_) => result[1] += 1,
                State::Recovered => result[2] += 1
            }
        }
        result
    }
}

