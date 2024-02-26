use std::cmp;

pub fn diff(x: &Vec<f64>, d: usize) -> Vec<f64> {
    let mut y: Vec<f64> = x.to_vec();
    let len = y.len();
    for s in 0..d {
        for i in 1..len - s {
            // we iterate backwards through the vector to avoid cloning
            y[len - i] = y[len - i] - y[len - i - 1];
        }
    }
    y.drain(0..d);
    y
}

pub fn inverse_diff(x: &Vec<f64>, d: usize) -> Vec<f64> {
    let y: Vec<f64> = vec![0.0; d];
    let mut cum: Vec<f64> = vec![y, x.to_vec()].concat().to_vec();

    for _ in 0..d {
        cum = cumsum(cum);
    }
    cum
}

pub fn cumsum(x: Vec<f64>) -> Vec<f64> {
    let mut y: Vec<f64> = Vec::new();
    if x.len() < 2 {
        y.push(From::from(0));
        return y;
    }
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(y[i - 1] + x[i]);
    }
    y
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

pub fn compute_variance(data: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {   
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