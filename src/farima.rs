use super::utils::{mean, diff, pacf, diffseries, residuals, closest_integer, compute_variance};
use liblbfgs::lbfgs;
use finitediff::FiniteDiff;

/// FARIMA struct represents a fractional autoregressive integrated moving average model.
#[derive(Debug, Clone)]
pub struct FARIMA {
    pub phi: Vec<f64>,          // AR coefficients
    pub diff: f64,              // Fractional differencing parameter
    pub theta: Vec<f64>,        // MA coefficients
    pub sigma_squared: f64      // Variance of the model
}

/// FARIMAMethod represents different methods for fitting a FARIMA model.
pub enum FARIMAMethod {}

impl FARIMA {
    /// Creates a new FARIMA struct with default values.
    pub fn new() -> FARIMA {
        let phi: Vec<f64> = vec![0.0; 1];
        let theta: Vec<f64> = vec![0.0; 1];
        FARIMA { phi, diff: 0.0, theta, sigma_squared: 0.0 }
    }

    /// Prints a summary of the FARIMA model.
    pub fn summary(&self) {
        println!("phi: {:?}\nd: {}\ntheta: {:?} \nsigma_squared {}", self.phi, self.diff, self.theta, self.sigma_squared);
    }

    /// Fits the FARIMA model to the provided data.
    pub fn fit(&mut self, data: &Vec<f64>, p: usize, d: f64, q: usize) {

        let int_d = closest_integer(d);

        // Fractional integration
        let mut diff_data = diffseries(data, d - int_d as f64);
        diff_data = diff(&diff_data, int_d);

        self.diff = d;
        Self::fit_css(self, &diff_data, p, q);
        self.sigma_squared = compute_variance(&diff_data, &self.phi);
    }

    fn fit_css(&mut self, data: &Vec<f64>, p: usize, q: usize) {

        let total_size = 1 + p + q;

        // The objective is to minimize the conditional sum of squares (CSS),
        // i.e. the sum of the squared residuals
        let f = |coef: &Vec<f64>| {
            assert_eq!(coef.len(), total_size);

            let intercept = coef[0];
            let phi = &coef[1..p + 1];
            let theta = &coef[p + 1..];

            let residuals = residuals(&data, intercept, &phi.to_vec(), &theta.to_vec());

            let mut css: f64 = 0.0;
            for residual in &residuals {
                css += residual * residual;
            }
            css
        };
        let g = |coef: &Vec<f64>| coef.forward_diff(&f);

        // Initial coefficients
        let mut coef: Vec<f64> = Vec::new();

        // Initial guess for the intercept: First value of data
        coef.push(mean(&data));

        // Initial guess for the p coefficients: Values of the PACF
        if p > 0 {
            let pacf = pacf(&data, Some(p));
            for p in pacf {
                coef.push(p);
            }
        }

        // Initial guess for the MA coefficients: 1.0
        if q > 0 {
            coef.resize(coef.len() + q, 1.0);
        }

        let evaluate = |x: &[f64], gx: &mut [f64]| {
            let x = x.to_vec();
            let fx = f(&x);
            let gx_eval = g(&x);
            // copy values from gx_eval into gx
            gx[..gx_eval.len()].copy_from_slice(&gx_eval[..]);
            Ok(fx)
        };

        let fmin = lbfgs().with_max_iterations(200);
        if let Err(_e) = fmin.minimize(
            &mut coef, // input variables
            evaluate,  // define how to evaluate function
            |_prng| {
                false 
            },
        ) {}

        self.phi = coef[1..=p].to_vec();
        self.theta = coef[p+1..].to_vec();
    }
}
