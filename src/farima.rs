use super::utils::{mean, diff, pacf, diffseries, residuals, closest_integer, compute_variance};
use liblbfgs::lbfgs;
use finitediff::FiniteDiff;

#[derive(Debug, Clone)]
pub struct FARIMA {
    phi: Vec<f64>,
    diff: f64,
    theta: Vec<f64>,
    sigma_squared: f64
}

pub enum FARIMAMethod {}

impl FARIMA {
    pub fn new() -> FARIMA {
        let phi: Vec<f64> = vec![0.0; 1];
        let theta: Vec<f64> = vec![0.0; 1];
        FARIMA { phi, diff: 0.0, theta, sigma_squared: 0.0 }
    }

    pub fn summary(&self) {
        println!("phi: {:?}\nd: {}\ntheta: {:?} \nsigma_squared {}", self.phi, self.diff, self.theta, self.sigma_squared);
    }

    // pub fn simulate(
    //     &self,
    //     length: usize,
    //     phi: Vec<f64>,
    //     d: f64,
    //     theta: Vec<f64>,
    //     error_mean: f64,
    //     error_variance: f64,
    // ) -> Vec<f64> {
    //     let mut output: Vec<f64> = Vec::with_capacity(length);
    //     output
    // }

    pub fn fit(&mut self, data: &Vec<f64>, p: usize, d: f64, q: usize) {

        let int_d = closest_integer(d);
        // println!("d f {}", d - int_d as f64);
        // fractional integration
        let mut diff_data = diffseries(data, d - int_d as f64);
        // println!("fint {:?}",diff_data);
        // println!("------------");
        // integer integration
        diff_data = diff(&diff_data, int_d);
        // println!("int {:?}",diff_data);

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
        // Todo: These initial guesses are rather arbitrary.
        let mut coef: Vec<f64> = Vec::new();

        // Initial guess for the intercept: First value of data
        coef.push(mean(&data));

        // println!("media: {}", mean(&data));

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
        ) {
            // tracing::warn!("{}", e);
        }

        // println!("cola{:?}", coef);

        
        self.phi = coef[1..=p].to_vec();
        self.theta = coef[p+1..].to_vec();
    }
}

