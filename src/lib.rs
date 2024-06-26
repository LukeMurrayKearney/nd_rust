use distributions::median;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::dpln::{pdf, sample};
use crate::run_model::abc_r0;

mod network_structure;
mod distributions;
mod dpln;
mod connecting_stubs;
mod network_properties;
mod run_model;

////////////////////////////////////////////// Network Creation ////////////////////////////////////////

//  Creates a network from given variables
#[pyfunction]
fn network_from_vars(n: usize, partitions: Vec<usize>, dist_type: &str, params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &params, &contact_matrix);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

// Creates a SBM network
#[pyfunction]
fn sbm_from_vars(n: usize, partitions: Vec<usize>, contact_matrix: Vec<Vec<f64>>) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}


/////////////////////////////////////////////// R_0 fitting /////////////////////////////////////////////

#[pyfunction]
fn test_r0_fit(n: usize, partitions: Vec<usize>, dist_type: &str, params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>, parameters: Vec<f64>, prop_infec: f64, num_networks: usize, target_r0: f64, iters: usize, num_replays: usize, scaling: &str) -> PyResult<Py<PyDict>> {
    
    let mut taus = Vec::new();
    for _ in 0..num_networks {
        let network: network_structure::NetworkStructure = match dist_type { 
            "sbm" => {
                network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix)
            },
            _ => network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &params, &contact_matrix)
        };
        let mut properties = network_properties::NetworkProperties::new(&network, &parameters);
        taus.push(abc_r0(&network, &mut properties, prop_infec, target_r0, iters, num_replays, false, scaling));
    }
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("tau", median(&mut taus).to_object(py))?;
        dict.set_item("taus_distribution", taus.to_object(py))?;
        
        return Ok(dict.into())
    })
}

//////////////////////////////////////////// outbreak simulation //////////////////////////////////////


#[pyfunction]
fn infection_sims(iters: usize, n: usize, partitions: Vec<usize>, dist_type: &str, params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>, parameters: Vec<f64>, maxtime: usize, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>>{
    
    let mut infections: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new()]; partitions.len()];
    let mut final_sizes: Vec<usize> = Vec::new();
    let mut peak_times: Vec<usize> = Vec::new();
    // A vector of each generation, with a vector of secondary cases from each iteration 
    let mut secondary_cases_vec: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new()]];
    // same as secondary cases for degrees
    let mut degrees_vec: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new()]];

    for _ in 0..iters {
        let network: network_structure::NetworkStructure = match dist_type { 
            "sbm" => {
                network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix)
            },
            _ => network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &params, &contact_matrix)
        };
        let mut properties = network_properties::NetworkProperties::new(&network, &parameters);
        let (individuals, seir, generation, secondary_cases, _) = run_model::run_tau_leap(&network, &mut properties, maxtime, prop_infec, scaling);
        // return final size of iteration
        peak_times.push(seir.iter().enumerate().max_by_key(|&(_, inner_vec)| inner_vec[2]).map(|(index, _)| index).unwrap());
        final_sizes.push(*seir.last().unwrap().last().unwrap());
        // get secondary cases and degrees in each generation
        let final_gen = *generation.iter().max().unwrap();
        for g in 0..final_gen {
            let indices: Vec<usize> = generation.iter().enumerate().filter(|(_, &val)| val == g).map(|(i, _)| i).collect();
            if g < secondary_cases_vec.len() {
                secondary_cases_vec[g].push(indices.iter().map(|&i| secondary_cases[i]).collect());
                degrees_vec[g].push(indices.iter().map(|&i| network.degrees[i]).collect());
            }
            else {
                secondary_cases_vec.push(vec![indices.iter().map(|&i| secondary_cases[i]).collect()]);
                degrees_vec.push(vec![indices.iter().map(|&i| network.degrees[i]).collect()]);
            }
        }


        // calculate infections at each time step for each age group
        for (t, people) in individuals.iter().enumerate() {
            for i in 0..partitions.len() {
                if infections[i].len() <= t {
                    infections[i].push(Vec::new());
                }
                infections[i][t].push(0);
            }
            for (i, _) in people.iter().enumerate().filter(|(_, state)| match state {1 => true, _ => false}) {
                let mut last = 0;
                match infections[network.ages[i]][t].len() {
                    0 => infections[network.ages[i]][t].push(0),
                    _ => {last = infections[network.ages[i]][t].len() - 1}
                };
                infections[network.ages[i]][t][last] += 1;
            }
        }
    }

    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        // Insert vectors into the results dictionary
        dict.set_item("infections", infections.to_object(py))?;
        dict.set_item("final sizes", final_sizes.to_object(py))?;
        dict.set_item("peak time", peak_times.to_object(py))?;
        dict.set_item("secondary cases by gen", secondary_cases_vec.to_object(py))?;
        dict.set_item("degrees by gen", degrees_vec.to_object(py))?;
        dict.set_item("tau", parameters[0].to_object(py))?;
        

        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

//////////////////////////////// Double Pareto Log-Normal functions /////////////////////////////////////////

#[pyfunction]
pub fn fit_dpln(data: Vec<f64>, iters: usize, prior_params: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        // Attempt to run the optimization
        match dpln::fit_dpln(data, iters, prior_params) {
            Ok(params) => {
                dict.set_item("alpha", params.alpha.to_object(py))?;
                dict.set_item("beta", params.beta.to_object(py))?;
                dict.set_item("nu", params.nu.to_object(py))?;
                dict.set_item("tau", params.tau.to_object(py))?;
                return Ok(dict.into());
            }, // If everything is okay, return Ok(())
            Err(ArgminError) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                ArgminError.to_string()
            )),
        }
    })
}

#[pyfunction]
pub fn dpln_sample(params: Vec<f64>, n: usize) -> Vec<f64> {

    sample(params, n)
}

#[pyfunction]
pub fn dpln_pdf(xs: Vec<f64>, params: Vec<f64>) -> Vec<f64> {

    pdf(xs, params)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn nd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(network_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(sbm_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(test_r0_fit, m)?)?;
    m.add_function(wrap_pyfunction!(infection_sims, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_sample, m)?)?;
    m.add_function(wrap_pyfunction!(fit_dpln, m)?)?;
    Ok(())
}