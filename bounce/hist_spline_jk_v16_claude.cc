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
#include <omp.h>


#include "header.h"

// O(3) thermal bounce action S = 4\pi \int r^2 ( (dq/dr)^2 + V ) dr (trapezoid on the shot
// trajectory). Factored out so the (B) dq rescan can reject a traversed-but-negative-action
// (wrong-vacuum / overshoot) trajectory the same way the final accept does.
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
#ifdef _OPENMP
  std::cout << "# nparallel = " << std::endl;
#endif


  // --------------------

  const int nptsx = 40; // sig0p5 32c: N_s=32 -> npts=40 (24c clean-room was 32) // 24c: N_s=24 -> npts=32 (32c was 40)
  // sig0p5/24c (stale): const int nptsy = 32;
  const int nptsy = 80; // v16: ImL bins DOUBLED; realaxis central row = nptsy/2-1 = 39 (data 32b_v16)
  const double minOx = -0.2;
  const double maxOx = 0.6;

  const std::string basedir = "/mnt/hdd_barracuda/llnl/reweight/data/32b_v16/"; // v16 (was stale 32b_v15)

  std::string mass = "0p4000";
  if(argc>=5) mass = argv[4];

  const int Nt=8;
  const int Ns=32; // 24c
  assert(Ns==32);


  std::vector<double> xpts(nptsx);
  const double deltax = (maxOx-minOx)/nptsx;
  for(int iptx=0; iptx<nptsx; iptx++) xpts[iptx] = minOx + deltax*(iptx+0.5);

  std::vector<double> betas;
  {
    const std::string filename = basedir + "/m"+mass+"avghist_ibx"+std::to_string(0)+"_iby"+std::to_string(nptsy/2-1)+"_nojkmeas.bin"; // v16: central realaxis row = nptsy/2-1 = 39 (was nptsx/2-1=19 when nptsy=nptsx=40)
    const HighFive::File f(filename.c_str(), HighFive::File::ReadOnly);
    betas = f.getDataSet("beta").read<std::vector<double>>();
  }
  const int nbeta_meas = betas.size();

  int nbeta;
  if(argc>=8) nbeta=atoi(argv[7]);
  const int nbin=40;

  // optional resample-range args for per-resample tmax escalation (NO DROP): re-run ONLY a
  // targeted (jdrop,ibin) sub-block so a single failing resample can get its own sufficient tmax
  // without disturbing the converged ones (each writes its own S3_{ibeta0}.h5). Defaults = full.
  int jdrop_lo = 0;
  int jdrop_hi = nbeta;
  int ibin_lo  = 0;
  int ibin_hi  = nbin;
  if(argc>=9)  jdrop_lo = atoi(argv[8]);
  if(argc>=10) jdrop_hi = atoi(argv[9]);
  if(argc>=11) ibin_lo  = atoi(argv[10]);
  if(argc>=12) ibin_hi  = atoi(argv[11]);

  int ibetac0;
  { // read ibetac0
    const std::string dir = "./";
    std::ifstream file(dir+"betac_ibetac_mass"+mass+"_32.dat"); // v16: Ns=32 suffix (was stale _24)
    std::string str;
    double tmp;
    while (std::getline(file, str)){
      std::istringstream iss(str);
      iss >> tmp;
    }
    ibetac0 = int(tmp);
  }


  std::vector<std::vector<int>> ibetac_jk;
  { // read ibetac
    const std::string dir = "./";
    std::ifstream file(dir+"ibetac_jk_mass"+mass+"_32.dat"); // v16: Ns=32 suffix (was stale _24)
    std::string str;
    while (std::getline(file, str)){
      std::vector<int> ibetacs;
      std::istringstream iss(str);
      double v;
      while( iss >> v ) ibetacs.push_back( int(v) );
      ibetac_jk.push_back(ibetacs);
    }
  }
  assert( ibetac_jk.size()==nbeta );
  assert( ibetac_jk[0].size()==nbin );

  const std::string dir0 = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"/";

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(2)
// #endif
  for(int jdrop=jdrop_lo; jdrop<jdrop_hi; jdrop++){
    for(int ibin=ibin_lo; ibin<ibin_hi; ibin++){
  // int jdrop=3;
  // int ibin=17;
      const std::string desc = "jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
      const std::string dir = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"_"+desc+"/";
      const std::string filename1 = dir + "/gamma.h5";
      HighFive::File f1(filename1.c_str(), HighFive::File::ReadOnly );

      std::cout << "@@@@@@@@@@ jdrop = " << jdrop << " ibin = " << ibin << std::endl;

      int ibeta0 =700;
      int expn=8;
      if(argc>=2) ibeta0 = atoi(argv[1]);
      if(argc>=3) expn = atoi(argv[2]);

      const int ibetac = ibetac_jk[jdrop][ibin];
      const int dibeta = ibetac - ibetac0;
      const int ibeta = ibeta0 + dibeta;
      std::cout << "# ibeta = " << ibeta
                << " ibetac = " << ibetac
                << " dibeta = " << dibeta
                << " ibetac0 = " << ibetac0 << std::endl;

      const std::string filename0 = dir0 + "/S3_"+std::to_string(ibeta0)+".h5";
      HighFive::File f0(filename0.c_str(), HighFive::File::ReadOnly );
      const std::string filename3 = dir + "/S3_"+std::to_string(ibeta0)+".h5";
      // Truncate (not OpenOrCreate): each S3_{ibeta0}.h5 holds exactly one {ibeta0} group and is
      // fully regenerated per call, so a fresh file is semantically identical to the 32c
      // OpenOrCreate+unlink, and it clobbers any stale/corrupt leftover (e.g. from a killed run)
      // instead of failing to reopen it.
      HighFive::File f3(filename3.c_str(), HighFive::File::Truncate );

      const double delta = 1.0*std::pow(10,-expn);
      std::vector<double> yy;
      std::string dataname0=std::to_string(ibeta); // renormalized
      yy = f1.getDataSet(dataname0).read<std::vector<double>>();

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
        assert( abs(xA-xB)>1.0e-12 );

        auto Veff_sub = [&](const double x) { return Veff(x) - Veff(xB); };

        double tmax=12.;
        if(argc>=4) tmax = atof(argv[3]);
        std::cout << "# tmax = " << tmax << std::endl;

        double dq_init; // = 0.125*std::pow(10,-6);
        std::string dataname=std::to_string(ibeta0);
        dq_init = f0.getDataSet("dq_init").read<double>();
        // v16 jk seed: nudge the MEAN converged dq slightly toward LARGER dq so the seed sits on the
        // OVERSHOOT side of each resample's (fluctuating) critical dq -> search_root descends correctly
        // instead of walking to 0. MIMIC THE ORIGINAL script (spline_cpp_eps/hist_spline_jk.cc:163):
        // a small multiplicative nudge dq_init *= 1.2 (the x3 bump was too large -- over-bumps ~4 orders
        // near Tc where mean dq ~1e-13 -> lands outside the thin window). (user guidance)
        // dq_init *= 3.0;   // OLD (too large a bump)
        dq_init *= 1.2;
        std::cout << "# dq_init (mean x1.2 nudge) = " << dq_init << std::endl;

        int iter_max=1000;
        if(argc>=6) iter_max = atoi(argv[5]);
        std::cout << "# iter_max = " << iter_max << std::endl;

        double tau=1.0e-4;
        if(argc>=7) tau = atof(argv[6]);
        std::cout << "# tau = " << tau << std::endl;

        std::vector<double> ts, qs;
        std::cout << "# debug. search" << std::endl;
        // "completed" bounce = the field TRAVERSED the barrier (start vacuum -> far vacuum),
        // i.e. |q(r_end) - q(0)| is a good fraction of the minima separation |xB - xA|.
        // (Direction-agnostic; distinguishes a real bounce from an undershoot that stays put.)
        const double dxAB = std::abs(xB - xA);
        const double dq_seed = dq_init; // MEAN seed -- capture BEFORE search_root overwrites dq_init
        double c = search_root( ts, qs, dq_init, delta, xA, xB, Veff_prime, tmax, tau, iter_max );

        // (B) per-resample bracket-find (NO DROP): the MEAN dq seed can sit in the undershoot
        // region, where search_root walks dq the WRONG way (cost<0 -> dq->0) and never reaches the
        // bracket. The bracket dq ranges ~0.2x..10x the mean across resamples (scan_dq_diag).
        // Re-seeding search_root at 25 multipliers is CORRECT but far too slow: a multiplier that
        // misses does a full 8000-iter grind. Instead do a CHEAP cost() scan (one integration per
        // point) to LOCATE a real bracket -- a cost sign change between two points that BOTH reach
        // a finite radius (rend>0, excluding the "released past the barrier" rend=0 artifact) --
        // then seed search_root ONCE at the geometric midpoint (it locks on in a few steps).
        // Accept a COMPLETED bounce: |c|<tol AND traversed AND Scl>0.
        const double tol_ = 1.0e-4;
        const long nfull_ = std::lround(tmax/tau);
        auto is_good = [&](const std::vector<double>& tv, const std::vector<double>& qv, const double cc){
          // STRICT: rode ALL the way to tmax (asymptotes to xB, q'->0), not a marginal overshoot.
          return std::abs(cc) < tol_
              && (long)qv.size()==nfull_
              && bounce_action(tv, qv, Veff_sub, tau) > 0.0;
        };
        if( !is_good(ts, qs, c) ){
          // ABSOLUTE dq range, NOT a multiple of the mean: near Tc the mean dq is tiny (~2e-6) but
          // the resample bracket sits at ~100x the mean (exponential dq sensitivity), far above any
          // [.,15x mean] window. dq is a release displacement from xA, physically in (0, xB-xA), so
          // scan [1e-6, 0.9|xB-xA|] on a log grid -- straddles every real crossing (deep ~5e-3 and
          // near-Tc ~2e-4 alike). (void dq_seed to keep it around for the debug print.)
          (void)dq_seed;
          // NS dense enough to STRADDLE narrow near-Tc brackets: those sit at very small dq
          // (~1e-6, the near-Tc tiny-dq regime) and can be only ~2-3% wide, so a coarse grid
          // catches them only by alignment luck (a resample fails where its identical-potential
          // neighbor passes). 600 pts over [1e-7, 0.9|xB-xA|] ~ 2.7% spacing -> reliably straddled.
          const int NS = 600;
          const double dqlo = 1.0e-7;
          const double dqhi = 0.9*dxAB;
          double prev_dq = 0.0, prev_cc = 0.0, prev_rend = 0.0;
          for(int k=0; k<=NS; k++){
            const double dqg = dqlo*std::pow(dqhi/dqlo, double(k)/NS);
            std::vector<double> tg, qg;
            const double cg = cost( tg, qg, xA+dqg, xA, xB, Veff_prime, tmax, tau );
            const double rg = tg.back();
            if( k>0 && prev_cc*cg < 0.0 && rg > 1.0e-6 && prev_rend > 1.0e-6 ){
              // real bracket in [prev_dq, dqg]: seed search_root at the geometric mid
              std::vector<double> ts2, qs2;
              double dqv = std::sqrt(prev_dq*dqg);
              // delta sized to the BRACKET dq, not the global delta (=10^-expn from the mean dq):
              // near Tc the mean dq is ~1e-13 -> expn=15 -> delta=1e-15 would grind at a bracket
              // that sits at ~2e-4. Use 10^-(floor(-log10(dqv))+2) so the bisection step matches dqv.
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

        // ceil(N/100) slots: with the relaxed solver break ts.size() is no longer a multiple
        // of 100 (bounce completes at r<tmax), so ts.size()/100 would undersize -> tsn[i/100]
        // writes past the end (heap corruption). (ts.size()+99)/100 = exact count of multiples.
        std::vector<double> tsn((ts.size()+99)/100), qsn((qs.size()+99)/100);
        for(std::size_t i=0; i<ts.size(); i++){
          if(i%100==0){
            tsn[i/100] = ts[i];
            qsn[i/100] = qs[i];
          }
        }

        std::cout << "# debug. ts.size() = " << ts.size() << std::endl;
        if (f3.exist(dataname+"/ts")) f3.unlink(dataname+"/ts");
        f3.createDataSet<std::vector<double>>(dataname+"/ts", tsn);
        if (f3.exist(dataname+"/qs")) f3.unlink(dataname+"/qs");
        f3.createDataSet<std::vector<double>>(dataname+"/qs", qsn);

        if (f3.exist(dataname+"/c")) f3.unlink(dataname+"/c");
        f3.createDataSet<double>(dataname+"/c", c);
        if (f3.exist(dataname+"/dq_init")) f3.unlink(dataname+"/dq_init");
        f3.createDataSet<double>(dataname+"/dq_init", dq_init);

        // bounce solution (same trapezoid as the (B) guard)
        const double sum = bounce_action(ts, qs, Veff_sub, tau);
        if (f3.exist(dataname+"/Scl")) f3.unlink(dataname+"/Scl");
        f3.createDataSet<double>(dataname+"/Scl", sum);

        // convergence flag (STRICT, the CORRECT bounce): CONVERGED iff |c|<tol, Scl>0, AND the
        // field rode ALL the way to tmax (ngot==nfull) -- it asymptotes to xB with q'->0. A
        // trajectory that reaches xB then escapes early (ngot<nfull) is a marginal OVERSHOOT
        // (q'!=0 at xB), biased high, NOT accepted. cause: !reached_far -> undershoot; Scl<=0 ->
        // bad_action; !rode_to_tmax -> escaped_early; else |c|>=tol -> unresolved_root.
        const double tol = 1.0e-4;
        const long nfull = std::lround(tmax/tau);
        const long ngot  = (long)ts.size();
        const bool rode_to_tmax = ( ngot == nfull );
        const bool reached_far = ( std::abs(qs.back()-qs.front()) > 0.4*dxAB ); // diagnostic
        const bool converged = ( std::abs(c) < tol ) && ( sum > 0.0 ) && rode_to_tmax;
        std::string cause = "converged";
        if(!converged){
          if( !reached_far ) cause = "undershoot";
          else if( sum <= 0.0 ) cause = "bad_action";
          else if( !rode_to_tmax ) cause = "escaped_early";
          else cause = "unresolved_root";
        }
        if (f3.exist(dataname+"/converged")) f3.unlink(dataname+"/converged");
        f3.createDataSet<int>(dataname+"/converged", converged?1:0);
        std::cout << "# CONVERGED " << ibeta0 << " " << (converged?1:0)
                  << " c=" << c << " nfull=" << nfull << " ngot=" << ngot
                  << " Scl=" << sum << " cause=" << cause
                  << " dq=" << dq_init << " qtrav=" << std::abs(qs.back()-qs.front())
                  << " jdrop=" << jdrop << " ibin=" << ibin << std::endl;

        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
      }
    }} // end for jk

  return 0;
}
