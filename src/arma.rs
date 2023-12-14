use std::collections::btree_map::Values;

use lstsq::Lstsq;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct ARMA {
    phi: Vec<f64>,
    theta: Vec<f64>,
}

pub enum ARMAMethod {
HRISSANEN,
}

impl ARMA {

    pub fn new() -> ARMA {
        let phi: Vec<f64> = vec![0.0; 1];
        let theta: Vec<f64> = vec![0.0; 1];
        ARMA { phi, theta }
    }

    pub fn summary(&self) {
        println!("phi: {:?}\nthetha: {:?}", self.phi, self.theta);
    }

    pub fn simulate(
        &self,
        length: usize,
        ar_param: Vec<f64>,
        ma_param: Vec<f64>,
        error_mean: f64,
        error_variance: f64,
    ) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(length);

        let ar_order = ar_param.len();
        let ma_order = ma_param.len();
        let normal: Normal<f64> = Normal::new(error_mean, error_variance.sqrt()).unwrap();

        let init = ar_order + ma_order;
        for _ in 0..(init + length) {
            let mut rng = rand::thread_rng();
            let err = normal.sample(&mut rng);
            output.push(err);
        }

        if ma_order > 0 {
            let ma = ma_param;
            let err = output.clone();

            for i in (ma_order)..(init + length) {
                for j in 0..ma_order {
                    output[i] += ma[j] * err[i - j - 1];
                }
            }
        }

        if ar_order > 0 {
            let ar = ar_param;

            for i in (ma_order + ar_order)..(init + length) {
                for j in 0..ar_order {
                    output[i] += ar[j] * output[i - j - 1];
                }
            }
        }

        output
    }

    pub fn fit(&mut self, data: &Vec<f64>, ar_order: usize, ma_order: usize, method: ARMAMethod) {
        match method {
            ARMAMethod::HRISSANEN => Self::fit_hannan_rissanen(self, data, ar_order, ma_order),
        }
    }

    fn fit_hannan_rissanen(&mut self, data: &Vec<f64>, p: usize, q: usize) {}
}