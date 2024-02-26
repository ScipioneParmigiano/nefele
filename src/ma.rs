use std::usize;

use super::ar::{ARMethod, AutoRegressive};
use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, Normal};
use finitediff::FiniteDiff;
use liblbfgs::lbfgs;
use crate::arma::mean;
use crate::arma::residuals;

#[derive(Debug, Clone)]
pub struct MovingAverage {
    pub theta: Vec<f64>,
    pub sigma_squared: f64,
    pub aic:f64,
    pub bic:f64
}

pub enum MAMethod {
    DURBIN,
    CSS
}

pub enum MACriterion {
    AIC,
    BIC,
}

impl MovingAverage {
    // create a new MA struct
    pub fn new() -> MovingAverage {
        MovingAverage {
            theta: vec![0.0; 1],
            sigma_squared: 0.0,
            aic: 0.0,
            bic:0.0
        }
    }

    pub fn summary(&self) {
        println!(
            "coefficients: {:?} \nsigma^2: {}",
            self.theta, self.sigma_squared
        )
    }

    // simulate a MA process
    pub fn simulate(
        &self,
        length: usize,
        param: Vec<f64>,
        error_mean: f64,
        error_variance: f64,
    ) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(length);

        let ma_order = param.len();
        let normal: Normal<f64> = Normal::new(error_mean, error_variance.sqrt()).unwrap();

        // initialization
        let init = ma_order;
        for _ in 0..(init + length) {
            let mut rng = rand::thread_rng();
            let err = normal.sample(&mut rng);
            output.push(err);
        }

        // MA(theta)
        if ma_order > 0 {
            let ma = &param;
            let err = output.clone();

            for i in (ma_order)..(init + length) {
                for j in 0..ma_order {
                    output[i] += ma[j] * err[i - j - 1];
                }
            }
        }

        output[init..].to_vec()
    }

    pub fn fit(&mut self, data: &Vec<f64>, order: usize, method: MAMethod) {
        match method {
            MAMethod::DURBIN => Self::fit_durbin(self, data, order),
            MAMethod::CSS => Self::fit_css(self, data, order)
        }

        self.sigma_squared = compute_variance(&data, &self.theta);
        self.aic = compute_aic(data.len(), self.sigma_squared, order);
        self.bic = compute_bic(data.len(), self.sigma_squared, order);
    }

    pub fn autofit(&mut self, data: &Vec<f64>, max_order: usize, criterion: MACriterion) {
        match criterion {
            MACriterion::AIC => Self::autofit_aic(self, data, max_order),
            MACriterion::BIC => Self::autofit_bic(self, data, max_order),
        }
    }

    fn fit_durbin(&mut self, data: &Vec<f64>, order: usize) {
        let m: usize= ((10*order * data.len()) as f64).ln().round() as usize;
        // println!("{}", m);
        let n = data.len() - m;

        // first step: estimate ar(m)
        let mut ar_m = AutoRegressive::new();
        ar_m.fit(data, m, ARMethod::YWALKER);

        let mut eps: Vec<f64> = Vec::new(); //vec![0.0; m];

        for i in m..data.len() {
            let mut prediction = 0.0;
            for (idx, &param) in ar_m.phi.iter().enumerate() {
                prediction += param * data[i - idx - 1];
            }
            let error = data[i] - prediction;

            eps.push(error);
        }

        let y_: Vec<f64> = data[m..]
            .iter()
            .zip(eps.iter())
            .map(|(&d, &e)| d - e)
            .collect();

            // second step: estimate ma parameters through least squares
        let mut x = DMatrix::from_element(n - order, order, 0.0);
        let y: Vec<f64> = y_[order..].iter().copied().collect();
        let y = DVector::from_vec(y);

        for row in 0..(n - order) {
            for col in 0..order {
                x[(row, col)] = eps[order - 1 + row - col];
            }
        }

        let result = (x.transpose() * &x).try_inverse().unwrap() * x.transpose() * y;
        self.theta = result.iter().cloned().collect();
    }

    fn fit_css(&mut self, data: &Vec<f64>, ma: usize) {

        let total_size = 1 + ma;

        // The objective is to minimize the conditional sum of squares (CSS),
        // i.e. the sum of the squared residuals
        let f = |coef: &Vec<f64>| {
            assert_eq!(coef.len(), total_size);

            let ar = 0;
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
        
        self.theta = coef[1..].to_vec();
    }

    fn autofit_aic(&mut self, data: &Vec<f64>, max_order: usize) {
        let mut aic:Vec<f64> = Vec::with_capacity(max_order);
        for order in 1..(max_order+1){
            Self::fit(self, data, order, MAMethod::DURBIN);
            aic.push(self.aic);
        }

        let min_order = aic
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index + 1) // Adding 1 to get position
        .unwrap_or(0);

        // println!("{:?}",min_order);
        Self::fit(self, data, min_order, MAMethod::DURBIN);
    }

    fn autofit_bic(&mut self, data: &Vec<f64>, max_order: usize) {
        let mut bic_values: Vec<f64> = Vec::with_capacity(max_order);

        for order in 1..=max_order {
            let mut ma_model = MovingAverage::new();
            ma_model.fit(data, order, MAMethod::DURBIN);
            bic_values.push(ma_model.bic);
        }

        let min_order = bic_values
            .iter()
            .enumerate()
            .min_by(|&(_, bic_a), &(_, bic_b)| bic_a.partial_cmp(bic_b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index + 1)
            .unwrap_or(0);

        self.fit(data, min_order, MAMethod::DURBIN);
    }
    
}

fn compute_variance(data: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {
    // let n = coefficients.len();
    // let mut errors: Vec<f64> = Vec::new();

    // // errors for the AR(n) model
    // for i in n..data.len() {
    //     let mut error = data[i];
    //     for j in 0..n {
    //         error -= coefficients[j] * data[i - j - 1];
    //     }
    //     errors.push(error);
    // }

    // // variance of errors
    // let variance: f64 = errors.iter().map(|&e| (e).powi(2)).sum::<f64>() / (errors.len()-coefficients.len()) as f64;

    // variance

    let q = 0; // coefficients.len();
    let n = data.len();

    let mut errors: Vec<f64> = vec![0.0; n];

    // Calculate errors using the MA model
    for i in coefficients.len()..n {
        let mut error = data[i];
        for j in 0..coefficients.len() {
            error -= coefficients[j] * data[i - j - 1];
        }
        errors[i] = error;
    }

    // Compute the variance of errors
    let sum_of_squares: f64 = errors.iter().skip(q).map(|&e| e * e).sum();
    let variance = sum_of_squares / (n - q) as f64;

    variance
}

fn compute_aic(n: usize, residual_sum_of_squares: f64, p: usize) -> f64 {
    let k = p; // Number of parameters (p autoregressive parameters)
    let aic = 2.0 * k as f64 + n as f64 * (residual_sum_of_squares / n as f64).ln();
    aic
}

fn compute_bic(n: usize, residual_sum_of_squares: f64, p: usize) -> f64 {
    let k = p; // Number of parameters (p autoregressive parameters)
    let bic = n as f64 * (residual_sum_of_squares / n as f64).ln() + k as f64 * (n as f64).ln();
    bic
}

