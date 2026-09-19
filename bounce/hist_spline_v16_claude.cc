#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>


#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include <Eigen/Dense>


#include "header.h"








// Trapezoid O(3) bounce action S3 = 4\pi \int r^2 [ (1/2)(dx/dr)^2 + V ] dr (same as the inline sum
// below; used by the bracket-find is_good check). Copied from hist_spline_jk (24c robustness).
static double bounce_action( const std::vector<double>& ts, const std::vector<double>& qs,
                             const std::function<double(const double)>& Veff_sub, const double tau ){
  double sum = 0.0;
  for(std::size_t i=0; i+1<ts.size(); i++){
    const double r = 0.5*(ts[i]+ts[i+1]);
    const double x = 0.5*(qs[i]+qs[i+1]);
    const double V = Veff_sub(x);
    const double dxdr = ( qs[i+1]-qs[i] )/tau;
    sum += r*r * tau * (dxdr*dxdr + V);
  }
  return 4.0*M_PI*sum;
}


int main(int argc, char* argv[]){
  // int main(){
  std::cout << std::scientific << std::setprecision(25);
  std::clog << std::scientific << std::setprecision(25);

  // --------------------

  const int nptsx = 40; // sig0p5 32c: N_s=32 -> npts=40 (24c clean-room was 32) // 24c: N_s=24 -> npts=32 (32c was 40)
  // sig0p5 (v15): int nptsy = nptsx;  (nptsy=40)
  int nptsy = 2*nptsx; // v16: ImL bins DOUBLED (nptsy=80); ReL (nptsx=40) unchanged -> irow1=39
  const double minOx = -0.2;
  const double maxOx = 0.6;

  const std::string basedir = "/mnt/hdd_barracuda/llnl/reweight/data/32b_v16/"; // v16 (was 32b_v15)

  const int irow1 = nptsy/2-1; // central realaxis row (=39 for nptsy=80): only rows 39,40 exist in 32b_v16
  std::string mass = "0p4000";
  // double mass_dummy=0.3;
  if(argc>=5) mass = argv[4];

  const int Nt=8;
  const int Ns=32; // 24c


  std::vector<double> xpts(nptsx);
  const double deltax = (maxOx-minOx)/nptsx;
  for(int iptx=0; iptx<nptsx; iptx++) xpts[iptx] = minOx + deltax*(iptx+0.5);

  std::vector<double> betas;
  {
    const std::string filename = basedir + "/m"+mass+"avghist_ibx"+std::to_string(0)+"_iby"+std::to_string(irow1)+"_nojkmeas.bin";
    const HighFive::File f(filename.c_str(), HighFive::File::ReadOnly);
    betas = f.getDataSet("beta").read<std::vector<double>>();
  }
  const int nbeta_meas = betas.size();

  int ibeta =700;
  int expn=8;
  if(argc>=2) ibeta = atoi(argv[1]);
  if(argc>=3) expn = atoi(argv[2]);
  const double delta = 1.0*std::pow(10,-expn);

  const std::string dir = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"/";
  const std::string filename1 = dir + "/gamma.h5";
  HighFive::File f1(filename1.c_str(), HighFive::File::ReadOnly );

  std::vector<double> yy;
  std::string dataname=std::to_string(ibeta);
  yy = f1.getDataSet(dataname).read<std::vector<double>>();

  const std::string filename3 = dir + "/S3_"+std::to_string(ibeta)+".h5";
  HighFive::File f3(filename3.c_str(), HighFive::File::Overwrite );

  {
    // interpolate
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, nptsx);
    gsl_spline_init(spline, xpts.data(), yy.data(), nptsx);

    auto Veff = [&](const double x) { return gsl_spline_eval(spline, x, acc); };
    auto Veff_prime = [&](const double x) { return gsl_spline_eval_deriv(spline, x, acc); };
    auto Veff_prime2 = [&](const double x) { return gsl_spline_eval_deriv2(spline, x, acc); };

    double xA, xB;
    get_local_minima( xA, xB, xpts, deltax, Veff_prime, Veff_prime2);

    auto Veff_sub = [&](const double x) { return Veff(x) - Veff(xB); };
    const double dxAB = std::abs(xB - xA); // minima separation (bracket-find range + diagnostics)

    double tmax=12.;
    if(argc>=4) tmax = atof(argv[3]);
    std::cout << "# tmax = " << tmax << std::endl;

    double dq_init = 0.125*std::pow(10,-6);
    if(argc>=6) dq_init = atof(argv[5]);
    std::cout << "# dq_init = " << dq_init << std::endl;

    int iter_max=1000;
    if(argc>=7) iter_max = atoi(argv[6]);
    std::cout << "# iter_max = " << iter_max << std::endl;

    double tau=1.0e-4;
    if(argc>=8) tau = atof(argv[7]);
    std::cout << "# tau = " << tau << std::endl;

    std::vector<double> ts, qs;
    std::cout << "# debug. search" << std::endl;
    double c = search_root( ts, qs, dq_init, delta, xA, xB, Veff_prime, tmax, tau, iter_max );

    // (B) bracket-find fallback (NO DROP, NOT a loosened tolerance): the seeded dq can sit in the
    // undershoot region, where search_root walks dq the WRONG way (cost<0 -> dq->0) and never reaches
    // the bracket (exactly the v16 vs production-seed mismatch). Do a CHEAP cost() scan over the
    // ABSOLUTE dq range [1e-7, 0.9|xB-xA|] on a log grid to LOCATE a real bracket (cost sign change
    // between two points that BOTH reach finite radius), then seed search_root ONCE at the geometric
    // midpoint. Accept only a STRICT completed bounce. Ported from hist_spline_jk (24c robustness).
    const double tol_ = 1.0e-4;
    const long nfull_ = std::lround(tmax/tau);
    auto is_good = [&](const std::vector<double>& tv, const std::vector<double>& qv, const double cc){
      return std::abs(cc) < tol_
          && (long)qv.size()==nfull_
          && bounce_action(tv, qv, Veff_sub, tau) > 0.0;
    };
    if( !is_good(ts, qs, c) ){
      const int NS = 800;
      const double dqlo = 1.0e-13; // v16 dq_crit reaches ~1e-10..1e-13 near betac (was 1e-7, too high -> missed the bracket)
      const double dqhi = 0.9*dxAB;
      double prev_dq = 0.0, prev_cc = 0.0, prev_rend = 0.0;
      for(int k=0; k<=NS; k++){
        const double dqg = dqlo*std::pow(dqhi/dqlo, double(k)/NS);
        std::vector<double> tg, qg;
        const double cg = cost( tg, qg, xA+dqg, xA, xB, Veff_prime, tmax, tau );
        const double rg = tg.back();
        if( k>0 && prev_cc*cg < 0.0 && rg > 1.0e-6 && prev_rend > 1.0e-6 ){
          std::vector<double> ts2, qs2;
          double dqv = std::sqrt(prev_dq*dqg);
          const double delta_b = std::pow(10.0, -(std::floor(-std::log10(dqv))+2.0));
          const double c2 = search_root( ts2, qs2, dqv, delta_b, xA, xB, Veff_prime, tmax, tau, iter_max );
          if( is_good(ts2, qs2, c2) ){
            ts = ts2;
            qs = qs2;
            c = c2;
            dq_init = dqv;
            break;
          }
        }
        prev_dq = dqg;
        prev_cc = cg;
        prev_rend = rg;
      }
    }
    std::cout << "# dq_init = " << dq_init << std::endl;
    std::cout << "# c = " << c << std::endl;

    // ceil(N/100): the relaxed search_root break (header.h) makes ts.size() a non-multiple of
    // 100 (bounce completes at r<tmax), so ts.size()/100 undersizes -> tsn[i/100] overruns.
    std::vector<double> tsn((ts.size()+99)/100), qsn((qs.size()+99)/100);
    for(std::size_t i=0; i<ts.size(); i++){
      if(i%100==0){
        tsn[i/100] = ts[i];
        qsn[i/100] = qs[i];
      }
    }
    if (f3.exist("ts")) f3.unlink("ts");
    f3.createDataSet<std::vector<double>>("ts", tsn);
    if (f3.exist("qs")) f3.unlink("qs");
    f3.createDataSet<std::vector<double>>("qs", qsn);

    if (f3.exist("c")) f3.unlink("c");
    f3.createDataSet<double>("c", c);
    if (f3.exist("dq_init")) f3.unlink("dq_init");
    f3.createDataSet<double>("dq_init", dq_init);

    // bounce solution
    double sum = 0.0;
    for(int i=0; i<ts.size()-1; i++) {
      const double r = 0.5*(ts[i]+ts[i+1]);
      const double x = 0.5*(qs[i]+qs[i+1]);
      const double V = Veff_sub(x);
      const double dxdr = ( qs[i+1]-qs[i] )/tau;
      const double dI = r*r * tau * (dxdr*dxdr + V);

      sum += dI;
    }
    sum *= 4.0*M_PI;
    if (f3.exist("Scl")) f3.unlink("Scl");
    f3.createDataSet<double>("Scl", sum);

    // convergence flag (STRICT, the CORRECT bounce): CONVERGED iff |c|<tol, Scl>0, AND the field
    // rode ALL the way to tmax without escaping the well (ngot==nfull) -- i.e. it asymptotes to xB
    // with q'->0. A trajectory that reaches xB then escapes early (ngot<nfull) is a marginal
    // OVERSHOOT (q'!=0 at xB), biased high, and is NOT accepted. Only genuine near-Tc machine-
    // precision betas fail this (critical dq below the ULP of xA+dq_init).
    const double tol = 1.0e-4;
    // dxAB already declared above (bracket-find range)
    const long nfull = std::lround(tmax/tau);
    const long ngot  = (long)ts.size();
    const bool rode_to_tmax = ( ngot == nfull );
    const bool reached_far = ( std::abs(qs.back()-qs.front()) > 0.4*dxAB ); // diagnostic only
    const bool converged = ( std::abs(c) < tol ) && ( sum > 0.0 ) && rode_to_tmax;
    std::string cause = "converged";
    if(!converged){
      if( !reached_far ) cause = "undershoot";
      else if( sum <= 0.0 ) cause = "bad_action";
      else if( !rode_to_tmax ) cause = "escaped_early";
      else cause = "unresolved_root";
    }
    if (f3.exist("converged")) f3.unlink("converged");
    f3.createDataSet<int>("converged", converged?1:0);
    std::cout << "# CONVERGED " << ibeta << " " << (converged?1:0)
              << " c=" << c << " nfull=" << nfull << " ngot=" << ngot
              << " Scl=" << sum << " cause=" << cause << std::endl;

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
  }


  return 0;
}
