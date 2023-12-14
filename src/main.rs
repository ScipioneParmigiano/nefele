pub mod ar;
pub mod arima;
pub mod arma;
pub mod ma;
pub mod farima;

use ar::*;
use arima::*;
use arma::*;
use ma::*;
use farima::*;

fn main() {
    //autoregressive
    {
        // let mut ar = AutoRegressive::new();
        // let sim: Vec<f64> = ar.simulate(10000, vec![-0.5, 0.3, 0.1], 0., 1.);
        // println!("{:?}", sim);
        // let sim = vec![];
        // ar.fit(&sim, 3, ARMethod::OLS);
        // ar.summary();
        // ar.fit(&sim, 3, ARMethod::YWALKER);
        // ar.summary();
        // ar.fit(&sim, 3, ARMethod::BURG);
        // ar.summary();

        // ar.autofit(&sim, 18, ARCriterion::AIC);
        // ar.summary();
        // ar.autofit(&sim, 19, ARCriterion::BIC);
        // ar.summary();
    }

    //moving average
    {
        // let mut ma: MovingAverage = MovingAverage::new();
        // let sim = ma.simulate(1000, vec![0.4, -0.2], 0., 1.0);
        // println!("\nsim: {:?} \n \n", sim);
        // ma.fit(&sim, 2, MAMethod::DURBIN);
        // println!("{:?}", ma.theta);
        // ma.autofit(&sim, 4, MACriterion::AIC);
        // ma.summary();
        // ma.autofit(&sim, 4, MACriterion::BIC);
        // ma.summary();
    }

    // arma
    {
        // let mut arma: ARMA = ARMA::new();
        // let sim = arma.simulate(1000, vec![0.6], vec![0.4], 0., 1.0);
        // println!("{:?}", sim);
        // arma.fit(&sim, 1,1, ARMAMethod::);
        // arma.summary();
    }

    // arima
    {
        // let mut arima: ARIMA = ARIMA::new();
        // let sim = arima.simulate(1000, vec![-0.2], 1, vec![0.3], 0., 1.);
        // println!("{:?}", sim);
        // arima.fit(&sim, 1, 1, 1);
        // arima.summary();
    }

    // farima
    {
        // let farima = FARIMA::new();
        // let sim = farima.simulate(100, vec![0.1],1.0,  vec![0.3], 0.0, 1.0);
        // println!("{:?}", sim);
    }
}
