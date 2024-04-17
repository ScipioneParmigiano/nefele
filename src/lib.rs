pub mod ar;
pub mod arima;
pub mod arma;
pub mod ma;
pub mod farima;
mod utils;

// use ar::*;
// use arima::*;
// use arma::*;
// use ma::*;
// use farima::*;
// use utils::*;

// fn main() {
//     //autoregressive
//     {
//         // let mut ar = AutoRegressive::new();
//         // let sim: Vec<f64> = ar.simulate(1000, vec![0.1], 0., 1.);
//         // ar.fit(&sim, 1, ARMethod::CSS);
//         // ar.summary();
//         // ar.fit(&sim, 1, ARMethod::OLS);
//         // ar.summary();
//         // ar.fit(&sim, 1, ARMethod::YWALKER);
//         // ar.summary();
//         // ar.fit(&sim, 1, ARMethod::BURG);
//         // ar.summary();
        
//         // println!("{:?}", sim);

//         // ar.autofit(&sim, 18, ARCriterion::AIC);
//         // ar.summary();
//         // ar.autofit(&sim, 19, ARCriterion::BIC);
//         // ar.summary();
//     }

//     //moving average
//     {
//         // let mut ma: MovingAverage = MovingAverage::new();
//         // let sim = ma.simulate(1000, vec![0.4, -0.2], 0., 1.0);
//         // ma.fit(&sim, 2, MAMethod::CSS);
//         // ma.summary();
//         // ma.fit(&sim, 2, MAMethod::DURBIN);
//         // ma.summary();
        
//         // println!("\nsim: {:?} \n \n", sim);
        
//         // ma.autofit(&sim, 10, MACriterion::AIC);
//         // ma.summary();
//         // ma.autofit(&sim, 10, MACriterion::BIC);
//         // ma.summary();
//     }

//     // arma
//     {
//         // let mut arma: ARMA = ARMA::new();
//         // let sim = arma.simulate(2000, vec![0.1], vec![0.2], 0., 1.0);
//         // arma.fit(&sim, 1,1, ARMAMethod::CSS);
//         // arma.summary();
        
//         // println!("{:?}", sim);

//         // arma.autofit(&sim, 2, 2, ARMACriterion::BIC);
//         // arma.summary();
//     }

//     // arima
//     {
//         // let mut arima: ARIMA = ARIMA::new();
//         // let sim = arima.simulate(40, vec![0.2], 1, vec![0.3], 0., 1.0);
//         // println!("{:?}", sim.len());
//         // arima.fit(&sim, 1, 1, 1, ARIMAMethod::CSS);
//         // arima.summary();
        
//         // println!("{:?}", sim);

//         // arima.autofit(&sim, 1, 2, 2, ARIMACriterion::BIC);
//         // arima.summary();
//     }

//     // farima
//     {
//         let mut arima = ARIMA::new();
//         let mut farima = FARIMA::new();
//         let sim = arima.simulate(10000, vec![0.4], 1, vec![-0.1], 0.0, 1.0);
//         // println!("{:?}", sim);
//         farima.fit(&sim, 1, 1., 1);
//         arima.fit(&sim, 1, 1, 1, ARIMAMethod::CSS);  
//         arima.summary();      
//         farima.summary();
//         // println!("{:?}", sim);
//     }

//     // let d =0.1;
//     // let mut x = vec![3., 2., 1.];
//     // x= diffseries(&inverse_diffseries(&x, d), d);
//     // println!("{:?}", x);
// }

