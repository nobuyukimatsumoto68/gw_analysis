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
#include <highfive/H5Easy.hpp>

#include <Eigen/Dense>




#include "header.h"
#include "obs.h"




const double epsilon=0.5;



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

  const std::string dir0 = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"/";
  const std::string filename0 = dir0 + "/gamma.h5";
  HighFive::File f0(filename0.c_str(), HighFive::File::ReadOnly );

  for(int jdrop=jdrop_lo; jdrop<jdrop_hi; jdrop++){
    std::cout << "# jdrop = " << jdrop << std::endl;

    using T=Eigen::RowVectorXd;
    std::vector<JackknifeSimp<T>> vec_obs(ibetamax-ibetamin+1, JackknifeSimp<T>(nbin, nptsx));

    for(int ibin=0; ibin<nbin; ibin++){
      Eigen::ArrayXXd Gamma_list;
      const std::string desc = "jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
      get_Gamma( Gamma_list, nbeta_meas, nptsx, nptsy, irow1, basedir, mass, Ns, Nt, desc );

      for(int ibeta=ibetamin; ibeta<=ibetamax; ibeta++){
        Eigen::MatrixXd yy(1, nptsx);
        yy = Gamma_list.block( ibeta, 0, 1, nptsx );
        vec_obs[ibeta-ibetamin].jack_avg[ibin] = yy;
      }
    }

    for(int ibeta=ibetamin; ibeta<=ibetamax; ibeta++){
      std::vector<double> Vs0_;
      std::string dataname0=std::to_string(ibeta);
      Vs0_ = f0.getDataSet(dataname0).read<std::vector<double>>();
      T Vs0(Vs0_.size());
      Vs0 = Eigen::Map<T>(Vs0_.data(), Vs0_.size());

      auto& obs = vec_obs[ibeta-ibetamin];
      obs.eps_trick(epsilon, Vs0);

      for(int ibin=0; ibin<nbin; ibin++){
        // const std::string desc = "jk_"+std::to_string(jdrop);
        const std::string desc = "jk_"+std::to_string(jdrop)+"_"+std::to_string(nbin)+"_"+std::to_string(ibin);
        const std::string dir = "./fit_params_"+std::to_string(Ns)+"c_m"+mass+"_"+desc+"/";
        std::filesystem::create_directory( dir );
        const std::string filename1 = dir + "/gamma.h5";
        HighFive::File f1(filename1.c_str(), HighFive::File::OpenOrCreate );

        const T& array = obs.jack_avg[ibin];
        std::vector<double> tmp(array.data(), array.data()+array.size());
        std::string dataname=std::to_string(ibeta);
        // full-range re-run over 0-1000 on an existing (windowed) gamma.h5: overwrite existing
        // datasets so createDataSet does not throw on the old window betas.
        if(f1.exist(dataname)) f1.unlink(dataname);
        f1.createDataSet<std::vector<double>>(dataname, tmp);
      }
    }


  } // for jk

  return 0;
}
