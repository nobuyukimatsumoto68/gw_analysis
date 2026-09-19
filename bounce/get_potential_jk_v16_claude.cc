#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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





int main(int argc, char* argv[]){
  // int main(){
  std::cout << std::scientific << std::setprecision(25);
  std::clog << std::scientific << std::setprecision(25);

  // --------------------

  std::string mass = "0p4000";
  if(argc>=2) mass = argv[1];

  int ibetamin, ibetamax;
  if(argc>=3) ibetamin = atoi(argv[2]);
  if(argc>=4) ibetamax = atoi(argv[3]);

  const int Nt=8;
  int Ns=32;
  if(argc>=5) Ns = atoi(argv[4]);

  int nptsx = 40;
  if(argc>=6) nptsx = atoi(argv[5]);
  // sig0p5 (v15): int nptsy = nptsx;  (nptsy=40)
  int nptsy = 2*nptsx; // v16: ImL bins DOUBLED (nptsy=80); ReL (nptsx=40) unchanged -> irow1=39

  int nbeta;
  if(argc>=7) nbeta=atoi(argv[6]);
  const int nbin=40;

  // optional jdrop range for 12-way (mass,jdrop) parallelization; each jdrop writes DISJOINT
  // fit_params_*_jk_{jdrop}_* dirs -> safe in parallel. Default = full [0,nbeta).
  int jdrop_lo=0, jdrop_hi=nbeta;
  if(argc>=8) jdrop_lo=atoi(argv[7]);
  if(argc>=9) jdrop_hi=atoi(argv[8]);


  const std::string basedir = "/mnt/hdd_barracuda/llnl/reweight/data/"+std::to_string(Ns)+"b_v16/"; // v16 (was b_v15)


  const double minOx = -0.2;
  const double maxOx = 0.6;

  const int irow1 = nptsy/2-1;



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

  for(int jdrop=jdrop_lo; jdrop<jdrop_hi; jdrop++){
    for(int ibin=0; ibin<nbin; ibin++){
      const std::string desc = "jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
      const std::string dir = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"_"+desc+"/";
      const std::string filename1 = dir + "/gamma.h5";
      HighFive::File f1(filename1.c_str(), HighFive::File::ReadOnly );
      const std::string filename2 = dir + "/Veff.h5";
      // HighFive::File f2(filename2.c_str(), HighFive::File::OpenOrCreate );
      HighFive::File f2(filename2.c_str(), HighFive::File::Overwrite ); // full-range re-run: fresh Veff.h5 per dir (opened once per (jdrop,ibin), all betas written below)

      for(int ibeta=ibetamin; ibeta<=ibetamax; ibeta++){
        std::cout << "# ibeta: " << ibeta << std::endl;
        std::string dataname=std::to_string(ibeta);

        std::vector<double> yy;
        yy = f1.getDataSet(dataname).read<std::vector<double>>();

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

        const int xsize = 1000;
        const double xmin= *std::min_element(xpts.begin(), xpts.end()); // *std::min(xpts); // -0.1;
        const double xmax= *std::max_element(xpts.begin(), xpts.end()); // *std::min(xpts); // -0.1;
        std::vector<double> xs(xsize+1);
        for(int i=0; i<=xsize; i++) xs[i] = (xmax-xmin)/xsize * i + xmin;
        std::vector<double> Vs;
        for(const double x : xs) Vs.push_back( Veff_sub(x) );

        std::vector<double> minima_data{ xA, xB, Veff(xA), Veff(xB) };

        f2.createDataSet<std::vector<double>>(dataname+"/xs", xs);
        f2.createDataSet<std::vector<double>>(dataname+"/Vs", Vs);
        f2.createDataSet<std::vector<double>>(dataname+"/minima_data", minima_data);

        gsl_spline_free(spline);
        gsl_interp_accel_free(acc);
      }
    }} // for jk

  return 0;
}
