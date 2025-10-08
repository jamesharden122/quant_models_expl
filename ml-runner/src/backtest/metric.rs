use polars::prelude::*;

fn to_f64_vec(s: &Series) -> PolarsResult<Vec<f64>> {
    let s = s.cast(&DataType::Float64)?;
    Ok(s.f64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<f64>>())
}

/// Sample mean and sample standard deviation (ddof=1)
fn mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if x.is_empty() { return (0.0, 0.0); }
    let mean = x.iter().copied().sum::<f64>() / n;
    let var = if x.len() > 1 {
        x.iter().map(|v| {
            let d = v - mean; d * d
        }).sum::<f64>() / (n - 1.0)
    } else { 0.0 };
    (mean, var.sqrt())
}

/// Sharpe ratio using excess returns x_t = r_t - r_f,t.
/// Returns annualized Sharpe using sqrt(ann_factor).
pub fn sharpe_ratio(r: &Series, r_f: Option<&Series>, ann_factor: f64) -> PolarsResult<f64> {
    let mut x = to_f64_vec(r)?;
    if let Some(rf) = r_f {
        let rf = to_f64_vec(rf)?;
        let n = x.len().min(rf.len());
        x.truncate(n);
        for i in 0..n { x[i] = x[i] - rf[i]; }
    }
    let (mu, sd) = mean_std(&x);
    if sd <= f64::EPSILON { return Ok(0.0); }
    Ok((mu / sd) * ann_factor.sqrt())
}

/// Sortino ratio using MAR_t as the minimal acceptable return (default 0).
/// Annualized by sqrt(ann_factor).
pub fn sortino_ratio(r: &Series, mar: Option<&Series>, ann_factor: f64) -> PolarsResult<f64> {
    let mut d = to_f64_vec(r)?; // d_t = r_t - MAR_t
    if let Some(m) = mar {
        let m = to_f64_vec(m)?; let n = d.len().min(m.len()); d.truncate(n);
        for i in 0..n { d[i] = d[i] - m[i]; }
    }
    let n = d.len();
    if n == 0 { return Ok(0.0); }
    let mean_d = d.iter().copied().sum::<f64>() / (n as f64);
    // downside deviation: sqrt( E[min(d,0)^2] ) with population denominator T
    let down_var = d.iter().map(|v| {
        let m = v.min(0.0); m * m
    }).sum::<f64>() / (n as f64);
    let down_sd = down_var.sqrt();
    if down_sd <= f64::EPSILON { return Ok(0.0); }
    Ok((mean_d / down_sd) * ann_factor.sqrt())
}

/// Maximum drawdown computed from cumulative wealth index V_t.
pub fn max_drawdown(r: &Series) -> PolarsResult<f64> {
    let x = to_f64_vec(r)?;
    let mut v = 1.0f64; // V_0
    let mut peak = 1.0f64;
    let mut mdd = 0.0f64;
    for ret in x {
        v *= 1.0 + ret;
        if v > peak { peak = v; }
        let dd = 1.0 - (v / peak);
        if dd > mdd { mdd = dd; }
    }
    Ok(mdd)
}

/// Information ratio vs benchmark b_t.
/// Annualized by sqrt(ann_factor).
pub fn information_ratio(r_p: &Series, r_b: &Series, ann_factor: f64) -> PolarsResult<f64> {
    let mut a = to_f64_vec(r_p)?; // active return a_t = r_p - r_b
    let b = to_f64_vec(r_b)?; let n = a.len().min(b.len()); a.truncate(n);
    for i in 0..n { a[i] = a[i] - b[i]; }
    let (mu, sd) = mean_std(&a);
    if sd <= f64::EPSILON { return Ok(0.0); }
    Ok((mu / sd) * ann_factor.sqrt())
}

/// t-statistic for the mean return: t = mu / (sd / sqrt(n))
pub fn t_statistic(r: &Series) -> PolarsResult<f64> {
    let x = to_f64_vec(r)?;
    if x.is_empty() { return Ok(0.0); }
    let (mu, sd) = mean_std(&x);
    if sd <= f64::EPSILON { return Ok(0.0); }
    Ok(mu / (sd / (x.len() as f64).sqrt()))
}

