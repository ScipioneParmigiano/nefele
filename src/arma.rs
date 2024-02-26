use rand_distr::{Distribution, Normal};
use liblbfgs::lbfgs;
use std::cmp;
use finitediff::FiniteDiff;


#[derive(Debug, Clone)]
pub struct ARMA {
    phi: Vec<f64>,
    theta: Vec<f64>,
    sigma_squared: f64,
    pub aic: f64,
    bic: f64
}

pub enum ARMAMethod {
    HRISSANEN,
    CSS,
    ML
}

pub enum ARMACriterion{
    AIC,
    BIC
}

impl ARMA {
    // create a new ARMA struct
    pub fn new() -> ARMA {
        let phi: Vec<f64> = vec![0.0; 1];
        let theta: Vec<f64> = vec![0.0; 1];
        ARMA { phi, theta, sigma_squared: 0.0, aic: 0.0, bic: 0.0 }
    }

    pub fn summary(&self) {
        println!("phi: {:?} \ntheta: {:?} \nsigma^2: {:?}", self.phi, self.theta, self.sigma_squared);
    }

    // simulate an ARMA process
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

        // initialization
        let init = ar_order + ma_order;
        for _ in 0..(init + length) {
            let mut rng = rand::thread_rng();
            let err = normal.sample(&mut rng);
            output.push(err);
        }

        // MA(theta)
        if ma_order > 0 {
            let ma = ma_param;
            let err = output.clone();

            for i in (ma_order)..(init + length) {
                for j in 0..ma_order {
                    output[i] += ma[j] * err[i - j - 1];
                }
            }
        }

        // AR(phi)
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
            ARMAMethod::CSS => Self::fit_css(self, data, ar_order, ma_order),
            ARMAMethod::ML => Self::fit_ml(self, data, ar_order, ma_order),
        }
    
        self.sigma_squared = compute_variance(&data, &self.phi);
        self.aic = compute_aic(data.len(), self.sigma_squared, ar_order + ma_order);
        self.bic = compute_bic(data.len(), self.sigma_squared, ar_order + ma_order);
    }

    pub fn autofit(&mut self, data: &Vec<f64>, max_ar_order: usize, max_ma_order: usize, criterion: ARMACriterion) {
        match criterion {
            ARMACriterion::AIC => Self::autofit_aic(self, data, max_ar_order, max_ma_order),
            ARMACriterion::BIC => Self::autofit_bic(self, data, max_ar_order, max_ma_order),
        }
    }

    fn fit_hannan_rissanen(&mut self, data: &Vec<f64>, ar: usize, ma: usize) {}

    fn fit_ml(&mut self, data: &Vec<f64>, ar: usize, ma: usize) {}

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

    fn autofit_aic(&mut self, data: &Vec<f64>, max_ar_order: usize, max_ma_order: usize) {
        let mut aic:Vec<f64> = Vec::with_capacity(max_ar_order * max_ma_order);
        for ar_order in 0..(max_ar_order+1){
            for ma_order in 0..(max_ma_order+1){
                Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
                aic.push(self.aic);
                // println!("{:?}:", aic);
                // println!("ar: {} \nma: {}\naic: {}", ar_order, ma_order, self.aic);
            }  
        }   

        let min_order = aic
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index) // Adding 1 to get position
        .unwrap_or(0);

        // println!("\n {} \n", min_order);

        let ar_order = ((min_order / max_ar_order) as f64).floor() as usize;
        let ma_order = min_order.rem_euclid(max_ma_order);

        // println!("{:?}",min_order);
        // println!("{}\n{}", ar_order, ma_order);

        Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
    }

    fn autofit_bic(&mut self, data: &Vec<f64>, max_ar_order: usize, max_ma_order: usize){
        let mut bic:Vec<f64> = Vec::with_capacity(max_ar_order * max_ma_order);
            for ar_order in 1..(max_ar_order+1){
                for ma_order in 1..(max_ma_order+1){
                Self::fit(self, data, ar_order,ma_order, ARMAMethod::CSS);
                bic.push(self.bic);}
            // }

            // let _min_order = bic
            // .iter()
            // .enumerate()
            // .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            // .map(|(index, _)| index + 1) // Adding 1 to get position
            // .unwrap_or(0);

            let ar_order =1;
            let ma_order =1;

            // println!("{:?}",min_order);
            Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
        }
    }
}


fn compute_variance(data: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {   
    let n = data.len();
    let q = 0; //coefficients.len();

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
    let sum_of_squares: f64 = errors.iter().skip(coefficients.len()).map(|&e| e * e).sum();
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

pub fn residuals(
    x: &Vec<f64>,
    intercept: f64,
    phi: &Vec<f64>,
    theta: &Vec<f64>,
) -> Vec<f64> {
    let zero: f64 = From::from(0.0);

    let mut residuals: Vec<f64> = Vec::new();
    for _ in 0..phi.len() {
        residuals.push(zero);
    }
    for t in phi.len()..x.len() {
        let mut xt: f64 = intercept;
        for j in 0..phi.len() {
            xt += phi[j] * x[t - j - 1];
        }
        for j in 0..cmp::min(theta.len(), t) {
            xt += theta[j] * residuals[t - j - 1];
        }
        residuals.push(x[t] - xt);
    }

    residuals
}

pub fn pacf(
    x: &Vec<f64>,
    max_lag: Option<usize>,
) -> Vec<f64> {
    // get autocorrelations
    let rho = acf(x, max_lag, false);
    let cov0 = acf(x, Some(0), true)[0];
    pacf_rho_cov0(&rho, cov0, max_lag)
}

pub fn acf(
    x: &Vec<f64>,
    max_lag: Option<usize>,
    covariance: bool,
) -> Vec<f64> {
    let max_lag = match max_lag {
        // if upper bound for max_lag is n-1
        Some(max_lag) => cmp::min(max_lag, x.len() - 1),
        None => x.len() - 1,
    };
    let m = max_lag + 1;

    let len_x_usize = x.len();
    let len_x: f64 = From::from(len_x_usize as u32);
    let sum: f64 = From::from(0.0);

    let sum_x: f64 = x.iter().fold(sum, |sum, &xi| sum + xi);
    let mean_x: f64 = sum_x / len_x;

    //let mut y: Vec<f64> = Vec::with_capacity(max_lag);
    let mut y: Vec<f64> = vec![From::from(0.0); m];

    for t in 0..m {
        for i in 0..len_x_usize - t {
            let xi = x[i] - mean_x;
            let xi_t = x[i + t] - mean_x;
            y[t] += (xi * xi_t) / len_x;
        }
        // we need y[0] to calculate the correlations, so we set it to 1.0 at the end
        if !covariance && t > 0 {
            y[t] = y[t] / y[0];
        }
    }
    if !covariance {
        y[0] = From::from(1.0);
    }
    y
}

pub fn pacf_rho_cov0(
    rho: &Vec<f64>,
    cov0: f64,
    max_lag: Option<usize>,
) -> Vec<f64> {
    let max_lag = match max_lag {
        // if upper bound for max_lag is n-1
        Some(max_lag) => cmp::min(max_lag, rho.len() - 1),
        None => rho.len() - 1,
    };
    let m = max_lag + 1;

    // build output vector
    let mut y: Vec<f64> = Vec::new();

    // calculate AR coefficients for each solution of order 1..max_lag
    for i in 1..m {
        let (coef, _var) = ar_dl_rho_cov(rho, cov0, Some(i));
        // we now have a vector with i items, the last item is our partial correlation
        y.push(coef[i - 1]);
    }
    y
}

fn ar_dl_rho_cov(
    rho: &Vec<f64>,
    cov0: f64,
    order: Option<usize>,
) -> (Vec<f64>, f64) {
    let order = match order {
        Some(order) => cmp::min(order, rho.len() - 1),
        None => rho.len() - 1,
    };

    // we need zero values more than once, so we'll use this helper var
    let zero = 0.0;
    let one = 1.0;

    // these vectors will hold the parameter values
    let mut phi: Vec<Vec<f64>> = vec![Vec::new(); order + 1];
    let mut var: Vec<f64> = Vec::new();

    // initialize zero-order estimates
    phi[0].push(zero);
    var.push(cov0);

    for i in 1..order + 1 {
        // first allocate values for the phi vector so we can use phi[i][i-1]
        for _ in 0..i {
            phi[i].push(zero);
        }

        // estimate phi_ii, which is stored as phi[i][i-1]
        // phi_i,i = rho(i) - sum_{k=1}^{n-1}(phi_{n-1,k} * rho(n-k) /
        //  (1 - sum_{k=1}^{n-1}(phi_{n-1,k} * rho(k))

        let mut num_sum = zero; // numerator sum
        let mut den_sum = one; // denominator sum

        for k in 1..i {
            let p = phi[i - 1][k - 1];
            num_sum += p * rho[i - k];
            den_sum += -p * rho[k];
        }

        let phi_ii = (rho[i] - num_sum) / den_sum;
        phi[i][i - 1] = phi_ii;

        var.push(var[i - 1] * (one - phi_ii * phi_ii));

        for k in 1..i {
            phi[i][k - 1] = phi[i - 1][k - 1] - phi[i][i - 1] * phi[i - 1][i - k - 1];
        }
    }

    (phi[order].clone(), var[order])
}


pub fn mean(x: &Vec<f64>) -> f64 {
    let zero: f64 = From::from(0_i32);
    let n: f64 = From::from(x.len() as i32);
    x.iter().fold(zero, |sum, &item| sum + item) / n
}