use crate::network_structure::NetworkStructure;
use rand::distributions::WeightedIndex;
use statrs::distribution::Poisson;
use rand_distr::Distribution;

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

    pub fn initialize_infection_degree(&mut self, network: &NetworkStructure, proportion_of_population: f64, inv_gamma: f64) {

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
        let poisson_infectious_period = Poisson::new(inv_gamma).unwrap();
        // define random number generator
        let mut rng = rand::thread_rng();
        
        //we want a weighted sampling of the population
        let probabilities: Vec<f64> = network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).ln()).collect();

        // weighted index of each individual
        let dist = WeightedIndex::new(probabilities).unwrap();
        let selected: Vec<usize> = (0..number_of_infecteds).map(|_| dist.sample(&mut rng)).collect();

        // infect selected individuals
        for &i in selected.iter() {
            self.nodal_states[i] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
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

