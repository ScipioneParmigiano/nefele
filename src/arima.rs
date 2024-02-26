use rand_distr::{Distribution, Normal};
use liblbfgs::lbfgs;
use finitediff::FiniteDiff;
use super::ancilla::{compute_variance, diff, inverse_diff, residuals, mean, pacf, };

#[derive(Debug, Clone)]

pub struct ARIMA {
    pub phi: Vec<f64>,
    pub diff: usize,
    pub theta: Vec<f64>,
    sigma_squared: f64,
}

pub enum ARIMAMethod {
    CSS
}

impl ARIMA {
    pub fn new() -> ARIMA {
        ARIMA { phi: vec![0.0;1], diff:0, theta:vec![0.0;1], sigma_squared: 0.0 }
    }

    pub fn summary(&self) {
        println!(
            "phi: {:?}\nd: {}\ntheta: {:?} \nsigma^2: {:?}",
            self.phi, self.diff, self.theta, self.sigma_squared
        )
    }

    pub fn simulate(&self, length: usize, phi: Vec<f64>,
        diff: usize,
        theta: Vec<f64>, error_mean: f64, error_variance: f64) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(length);

        let ar_order = phi.len();
        let ma_order = theta.len();
        let normal: Normal<f64> = Normal::new(error_mean, error_variance.sqrt()).unwrap();

        let init = ar_order + ma_order;
        for _ in 0..(init + length) {
            let mut rng = rand::thread_rng();
            let err = normal.sample(&mut rng);
            output.push(err);
        }

        if ma_order > 0 {
            let ma = &theta;
            let err = output.clone();

            for i in (ma_order)..(init + length) {
                for j in 0..ma_order {
                    output[i] += ma[j] * err[i - j - 1];
                }
            }

            for i in 0..ma_order {
                output[i] = 0.0
            }
        }

        if ar_order > 0 {
            let ar = &phi;

            for i in (ma_order + ar_order)..(init + length) {
                for j in 0..ar_order {
                    output[i] += ar[j] * output[i - j - 1];
                }
            }
        }

        if diff > 0 {
            output = inverse_diff(&output[init..output.len() - diff].to_vec(), diff);
        } else {
            output.drain(0..init);
        }

        output
    }

    pub fn fit(&mut self, data: &Vec<f64>, p: usize, d: usize, q: usize, method: ARIMAMethod) {

        if d > 0 {
            let diff_data = diff(data, d);

            match method {
                ARIMAMethod::CSS => Self::fit_css(self, &diff_data, p, q)
            }

            self.diff = d;
            self.sigma_squared = compute_variance(&diff_data, &self.phi);
        } else {
            match method {
                ARIMAMethod::CSS => Self::fit_css(self, &data, p, q)
            }
            self.sigma_squared = compute_variance(&data, &self.phi);
        }
    }

    fn fit_css(&mut self, data: &Vec<f64>, ar: usize, ma: usize) {

        let total_size = 1 + ar + ma;

        // The objective is to minimize the conditional sum of squares (CSS),
        // i.e. the sum of the squared residuals
        let f = |coef: &Vec<f64>| {
            assert_eq!(coef.len(), total_size);

            let intercept = coef[0];
            let phi = &coef[1..ar + 1];
            let theta = &coef[ar + 1..];

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

        // Initial guess for the AR coefficients: Values of the PACF
        if ar > 0 {
            let pacf = pacf(&data, Some(ar));
            for p in pacf {
                coef.push(p);
            }
        }

        // Initial guess for the MA coefficients: 1.0
        if ma > 0 {
            coef.resize(coef.len() + ma, 1.0);
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
        if let Err(e) = fmin.minimize(
            &mut coef, // input variables
            evaluate,  // define how to evaluate function
            |_prng| {
                false 
            },
        ) {
            tracing::warn!("{}", e);
        }
        
        self.phi = coef[1..=ar].to_vec();
        self.theta = coef[ar+1..].to_vec();
    }
}