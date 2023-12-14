use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct AutoRegressive {
    pub phi: Vec<f64>,
    pub sigma_squared: f64,
    aic:f64,
    bic:f64
}

pub enum ARMethod {
    OLS,
    YWALKER,
    BURG,
}

pub enum ARCriterion {
    AIC,
    BIC,
}

impl AutoRegressive {

    pub fn new() -> AutoRegressive {
        AutoRegressive {
            phi: vec![0.0; 1],
            sigma_squared: 0.0,
            aic: 0.0,
            bic:0.0
        }
    }

    pub fn summary(&self) {
        println!(
            "coefficients: {:?} \nsigma^2: {}",
            self.phi, self.sigma_squared
        )
    }

    pub fn simulate(
        &mut self,
        length: usize,
        param: Vec<f64>,
        error_mean: f64,
        error_variance: f64,
    ) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::with_capacity(length);
        let normal: Normal<f64> = Normal::new(error_mean, error_variance.sqrt()).unwrap();

        let ar_order = param.len();

        let init = ar_order;
        for _ in 0..(init + length) {
            let mut rng = rand::thread_rng();
            let err = normal.sample(&mut rng);
            output.push(err);
        }

        if ar_order > 0 {
            let ar = &param;

            for i in (ar_order)..(init + length) {
                for j in 0..ar_order {
                    output[i] += ar[j] * output[i - j - 1];
                }
            }
        }

        output[ar_order..].to_vec()
    }

    pub fn fit(&mut self, data: &Vec<f64>, order: usize, method: ARMethod) {
        match method {
            ARMethod::OLS => Self::fit_ols(self, data, order),
            ARMethod::YWALKER => Self::fit_yule_walker(self, data, order),
            ARMethod::BURG => Self::fit_burg(self, data, order),
        }

        self.sigma_squared = compute_variance(&data, &self.phi);
        self.aic = compute_aic(data.len(), self.sigma_squared, order);
        self.bic = compute_bic(data.len(), self.sigma_squared, order);
    }

    pub fn autofit(&mut self, data: &Vec<f64>, max_order: usize, method: ARCriterion) {
        match method {
            ARCriterion::AIC => Self::autofit_aic(self, data, max_order),
            ARCriterion::BIC => Self::autofit_bic(self, data, max_order),
        }
    }

    fn fit_ols(&mut self, data: &Vec<f64>, order: usize) {
        let n = data.len();

        if n <= order {
            panic!("Not enough data for the given order");
        }

        let mut x = DMatrix::zeros(n - order, order);
        for i in order..n {
            for j in 0..order {
                x[(i - order, j)] = data[i - j - 1];
            }
        }

        let y = DVector::from_iterator(n - order, data.iter().skip(order).cloned());

        let xtx = x.transpose() * &x;
        let xty = x.transpose() * &y;

        let chol = xtx.cholesky().expect("Cholesky decomposition failed");
        let coefficients = chol.solve(&xty);

        self.phi = coefficients.data.into();
    }

    fn fit_yule_walker(&mut self, data: &Vec<f64>, order: usize) {
        let n = data.len();

        let mut rho = DMatrix::<f64>::zeros(order, order);

        for i in 0..order {
            for j in 0..order {
                let mut sum = 0.0;
                for k in 0..(n - order) {
                    sum += data[k + i] * data[k + j];
                }
                rho[(i, j)] = sum / (n - order) as f64;
            }
        }

        let mut r = DVector::<f64>::zeros(order);

        for i in 0..order {
            let mut sum = 0.0;
            for k in 0..(n - order) {
                sum += data[k + i] * data[k + order];
            }
            r[i] = sum / (n - order) as f64;
        }

        if let Some(solution) = rho.clone().qr().solve(&r) {
            self.phi = solution.iter().rev().cloned().collect();
        }
    }

    fn fit_burg(&mut self, data: &Vec<f64>, order: usize) {

        let mut r: Vec<f64> = vec![0.0; order + 1];
        for k in 0..=order {
            for i in k..data.len() {
                r[k] += data[i] * data[i - k];
            }
            r[k] /= (data.len() - k) as f64;
        }

        let mut e: Vec<f64> = vec![0.0; order + 1];
        let mut a: Vec<f64> = vec![0.0; order + 1];

        e[0] = r[0];
        for i in 1..=order {
            let mut lambda = 0.0;
            let mut sum = 0.0;
            for j in 1..=i - 1 {
                sum += a[j] * r[i - j];
            }
            lambda = (r[i] - sum) / e[i - 1];

            a[i] = lambda;
            for j in 1..=i - 1 {
                a[j] = a[j] - lambda * a[i - j];
            }

            e[i] = (1.0 - lambda * lambda) * e[i - 1];
        }

        self.phi = a[1..].to_vec();
    }

    fn autofit_aic(&mut self, data: &Vec<f64>, max_order: usize) {

        let mut aic:Vec<f64> = Vec::with_capacity(max_order);
        for order in 1..(max_order+1){
            Self::fit(self, data, order, ARMethod::YWALKER);
            aic.push(self.aic);
        }

        let min_order = aic
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index + 1) 
        .unwrap_or(0);

        Self::fit(self, data, min_order, ARMethod::YWALKER);
    }

    fn autofit_bic(&mut self, data: &Vec<f64>, max_order: usize) {
        let mut bic:Vec<f64> = Vec::with_capacity(max_order);
        for order in 1..(max_order+1){
            Self::fit(self, data, order, ARMethod::YWALKER);
            bic.push(self.bic);
        }

        let min_order = bic
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index + 1) 
        .unwrap_or(0);

        Self::fit(self, data, min_order, ARMethod::YWALKER);
    }

}

fn compute_variance(data: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {
    let n = coefficients.len();
    let mut errors: Vec<f64> = Vec::new();

    for i in n..data.len() {
        let mut error = data[i];
        for j in 0..n {
            error -= coefficients[j] * data[i - j - 1];
        }
        errors.push(error);
    }

    let variance: f64 = errors.iter().map(|&e| (e).powi(2)).sum::<f64>() / (errors.len()-coefficients.len()) as f64;

    variance
}

fn compute_aic(n: usize, residual_sum_of_squares: f64, p: usize) -> f64 {
    let k = p; 
    let aic = 2.0 * k as f64 + n as f64 * (residual_sum_of_squares / n as f64).ln();
    aic
}

fn compute_bic(n: usize, residual_sum_of_squares: f64, p: usize) -> f64 {
    let k = p; 
    let bic = n as f64 * (residual_sum_of_squares / n as f64).ln() + k as f64 * (n as f64).ln();
    bic
}


