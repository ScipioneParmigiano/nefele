use rand_distr::{Distribution, Normal};
use liblbfgs::lbfgs;
use finitediff::FiniteDiff;
use super::utils::{pacf, residuals, compute_aic, compute_bic, compute_variance, mean};

/// ARMA struct represents an autoregressive moving average model.
#[derive(Debug, Clone)]
pub struct ARMA {
    pub phi: Vec<f64>,              // AR coefficients
    pub theta: Vec<f64>,            // MA coefficients
    pub sigma_squared: f64,         // Variance of the model
    pub aic: f64,                   // AIC (Akaike Information Criterion) value
    pub bic: f64                    // BIC (Bayesian Information Criterion) value
}

/// ARMAMethod represents different methods for fitting an ARMA model.
pub enum ARMAMethod {
    CSS,    // Conditional Sum of Squares
    ML      // Maximum Likelihood
}

/// ARMACriterion represents criteria for selecting the order of the ARMA model.
pub enum ARMACriterion{
    AIC,    // Akaike Information Criterion
    BIC     // Bayesian Information Criterion
}

impl ARMA {
    /// Creates a new ARMA struct with default values.
    pub fn new() -> ARMA {
        let phi: Vec<f64> = vec![0.0; 1];
        let theta: Vec<f64> = vec![0.0; 1];
        ARMA { phi, theta, sigma_squared: 0.0, aic: 0.0, bic: 0.0 }
    }

    /// Prints a summary of the ARMA model.
    pub fn summary(&self) {
        println!("phi: {:?} \ntheta: {:?} \nsigma^2: {:?}", self.phi, self.theta, self.sigma_squared);
    }

    /// Simulates an ARMA process.
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

        // Initialization
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

    /// Fits the ARMA model to the provided data according to the selected method.
    pub fn fit(&mut self, data: &Vec<f64>, ar_order: usize, ma_order: usize, method: ARMAMethod) {
        match method {
            ARMAMethod::CSS => Self::fit_css(self, data, ar_order, ma_order),
            ARMAMethod::ML => Self::fit_ml(self, data, ar_order, ma_order),
        }

        
        self.sigma_squared = compute_variance(&data, &self.phi);
        self.aic = compute_aic(data.len(), self.sigma_squared, ar_order + ma_order);
        self.bic = compute_bic(data.len(), self.sigma_squared, ar_order + ma_order);
    }

    /// Automatically fits the ARMA model by selecting the order based on a criterion.
    pub fn autofit(&mut self, data: &Vec<f64>, max_ar_order: usize, max_ma_order: usize, criterion: ARMACriterion) {     
        match criterion {
            ARMACriterion::AIC => Self::autofit_aic(self, data, max_ar_order, max_ma_order),
            ARMACriterion::BIC => Self::autofit_bic(self, data, max_ar_order, max_ma_order),
        }
    }

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
        let mut aic: Vec<f64> = Vec::with_capacity((max_ar_order + 1) * (max_ma_order + 1));
    
        for ar_order in 0..=max_ar_order {
            for ma_order in 0..=max_ma_order {
                Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
                aic.push(self.aic);
            }
        }
    
        let min_order = aic
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);
    
        let ar_order = min_order / (max_ma_order + 1); // Integer division for ar_order
        let ma_order = min_order % (max_ma_order + 1); // Using modulo for ma_order
    
        Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
    }  

    fn autofit_bic(&mut self, data: &Vec<f64>, max_ar_order: usize, max_ma_order: usize){
        let mut bic:Vec<f64> = Vec::with_capacity(max_ar_order * max_ma_order);
            for ar_order in 1..(max_ar_order+1){
                for ma_order in 1..(max_ma_order+1){
                Self::fit(self, data, ar_order,ma_order, ARMAMethod::CSS);
                bic.push(self.bic);}

            let ar_order =1;
            let ma_order =1;

            Self::fit(self, data, ar_order, ma_order, ARMAMethod::CSS);
        }
    }
}
