#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <map>
#include <fstream>
#include <filesystem>
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_spline.h>

#include <omp.h>

// #include <highfive/highfive.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp> // per-jk dump (claude): one group per (MBGeV, MB) point
// #include <highfive/H5DataSet.hpp>
// #include <highfive/H5DataSpace.hpp>

// #include "H5Cpp.h"
// #include <highfive/H5DataSpace.hpp>


#include <Eigen/Dense>



using V=std::vector<double>;
using Vk=std::vector<V>;
using Vjk=std::vector<Vk>;
using Vjk_=std::vector<std::vector<std::vector<double>>>;

// #include "dof.h"
#include "dof_claude.h" // corrected gstar_HSDM = 44 (was 33.5): fermion term 7/8*4*Nc*Nf

// #include "header1.h"
#include "header1_v16_32c_c2only_claude.h" // adds g_v_wall: CLI-overridable wall velocity v (see header)
#include "obs.h"


const double epsilon = 0.5; // epscorrected (v16): eps_pot=0.5 (process_epsilon_v16), was 0.8 bug
const double eps = 0.5;
#define is_eps_trick true


void add_to( V& v, const double c0, const V& v0){
  for(int i=0; i<v.size(); i++) v[i] += c0*v0[i];
}

void rescale( V& v, const double& c0){
  for(int i=0; i<v.size(); i++) v[i] *= c0;
}


V get_mean( const Vk& v ){
  V res( v[0].size(), 0.0 );
  for(int i=0; i<v.size(); i++) {
    // res += v[i];
    add_to( res, 1.0, v[i]);
  }
  // res /= v.size();
  rescale( res, 1.0/v.size() );
  return res;
}


Vk eps_trick( const Vk& v,
              const double eps){
  const V mean = get_mean(v);

  Vk tmp = v;
  for(int i=0; i<v.size(); i++) add_to( tmp[i], -1.0, mean ); // tmp[i] -= mean;
  for(int i=0; i<v.size(); i++) rescale( tmp[i], eps );
  for(int i=0; i<v.size(); i++) add_to( tmp[i], 1.0, mean );
  return tmp;
}


// per-jk dump (claude): convert [njdrop][nbins] of the 8-vec GWParams into a plain
// [njdrop][nbins][nops] nested vector for HDF5, preserving (jdrop, ibin, op) indexing
// on the python side. GWParams is Eigen::Matrix<double,8,1>; nops=8 (matches main).
static Vjk_ to_nested_gwparams( const std::vector<std::vector<Eigen::Matrix<double,8,1>>>& jk ){
  Vjk_ out( jk.size() );
  for(size_t jd=0; jd<jk.size(); jd++){
    out[jd].resize( jk[jd].size() );
    for(size_t ib=0; ib<jk[jd].size(); ib++){
      out[jd][ib].resize( 8 );
      for(int op=0; op<8; op++) out[jd][ib][op] = jk[jd][ib](op);
    }
  }
  return out;
}


int main(int argc, char* argv[]){
  std::cout << std::scientific << std::setprecision(25);
  std::clog << std::scientific << std::setprecision(25);

  // wall velocity v from CLI: ./gw_fin_claude.o <v_w>
  // when v_w>0 it overrides the Chapman-Jouguet getv everywhere via g_v_wall
  // (both the Tstar percolation integral and the final vJ; see header1_claude.h).
  // no argument -> g_v_wall stays -1 -> physical CJ velocity vJ (reproduces gw_fin.cc).
  std::string psuf = "";
  if( argc > 1 ){
    psuf = argv[1];
    std::clog << "# param suffix = " << psuf << std::endl;
  }
  else {
    std::clog << "# using Chapman-Jouguet wall velocity vJ (g_v_wall = " << g_v_wall << ")" << std::endl;
  }


  // std::cout << gstar_SM(10000000) << std::endl;
  // for(double logT=std::log(TQCD); logT<std::log(500); logT+=0.01){
  //   const double T = std::exp(logT);
  //   std::cout << T << " " << gstar_SM(T) << std::endl;
  // }
  // return 1;

  // --------------------
  // std::ofstream ofs_freq_Omega("freq_Omega.dat");
  // ofs_freq_Omega << std::scientific << std::setprecision(25);
  // std::ofstream ofs_alpha_beta("alpha_beta.dat");
  // ofs_alpha_beta << std::scientific << std::setprecision(25);
  // std::ofstream ofs_freq_Omega("freq_Omega.dat");
  // ofs_freq_Omega << std::scientific << std::setprecision(25);


  if( argc > 2 ) g_v_wall = std::atof( argv[2] );   // fixed v_w (else vJ)
  V Tcparams;
  Vjk jk_Tcparams_m0p2000, jk_Tcparams_m0p3000, jk_Tcparams_m0p4000;
  {
    const std::string filename = "/mnt/hdd_barracuda/llnl/reweight/32c_reanalysis/alpha_beta/Tcparams" + psuf + ".hdf5";
    const HighFive::File f(filename.c_str(), HighFive::File::ReadOnly);
    Tcparams = f.getDataSet("mean").read<V>();
    jk_Tcparams_m0p2000 = f.getDataSet("jk_m0p2000").read<Vjk>();
    jk_Tcparams_m0p3000 = f.getDataSet("jk_m0p3000").read<Vjk>();
    jk_Tcparams_m0p4000 = f.getDataSet("jk_m0p4000").read<Vjk>();
  }
  V DeltaVparams;
  Vjk jk_DeltaVparams_m0p2000, jk_DeltaVparams_m0p3000, jk_DeltaVparams_m0p4000;
  {
    const std::string filename = "/mnt/hdd_barracuda/llnl/reweight/32c_reanalysis/alpha_beta/DeltaVparams" + psuf + ".hdf5";
    const HighFive::File f(filename.c_str(), HighFive::File::ReadOnly);
    DeltaVparams = f.getDataSet("mean").read<V>();
    jk_DeltaVparams_m0p2000 = f.getDataSet("jk_m0p2000").read<Vjk>();
    jk_DeltaVparams_m0p3000 = f.getDataSet("jk_m0p3000").read<Vjk>();
    jk_DeltaVparams_m0p4000 = f.getDataSet("jk_m0p4000").read<Vjk>();
  }
  V S3params;
  Vjk jk_S3params_m0p2000, jk_S3params_m0p3000, jk_S3params_m0p4000;
  {
    const std::string filename = "/mnt/hdd_barracuda/llnl/reweight/32c_reanalysis/alpha_beta/S3params" + psuf + ".hdf5";
    const HighFive::File f(filename.c_str(), HighFive::File::ReadOnly);
    S3params = f.getDataSet("mean").read<V>();
    jk_S3params_m0p2000 = f.getDataSet("jk_m0p2000").read<Vjk>();
    jk_S3params_m0p3000 = f.getDataSet("jk_m0p3000").read<Vjk>();
    jk_S3params_m0p4000 = f.getDataSet("jk_m0p4000").read<Vjk>();
  }


  // using GWParams=Eigen::Vector2d;
  constexpr int nops=8;
  using GWParams=Eigen::Matrix< double, nops, 1 >; // Eigen::VectorXd;

  const int NTs=2000;
  // const int NTs=200;
  const double threshold = -0.0006;
  const double threshold2 = 0.0006;
  std::vector<std::vector<double>> results;

  // for(const double MBGeV : std::vector<double>{5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0}){
  //for(const double MBGeV : std::vector<double>{20.0, 50.0, 100.0, 200.0, 500.0, 800.0}){
  std::vector<double> MBGeVs = {20.0, 70.0, 200.0, 500.0};
  std::vector<double> MBs    = {2.8, 3.2, 3.6, 4.0};
  if( argc > 3 && std::atof(argv[3]) > 0.0 ) MBGeVs = { std::atof(argv[3]) };  // single MBGeV (parallel split)
  if( argc > 4 && std::atof(argv[4]) > 0.0 ) MBs    = { std::atof(argv[4]) };  // single MB*sqrt(t0) (v_w scan)

  // per-jk dump (claude): open one HDF5 file for this process (one MBGeV, 4 MBs).
  // path from argv[5]; each (MBGeV, MB) point -> one group "pt_<i>". Truncate = fresh
  // per-process output (analogous to the "> part.dat" stdout redirect); re-run guarded
  // by the launcher flag. Reduction is NOT done here -- the offline python replicates it.
  std::string dumppath = "gw_jkdump_v16_32c_claude.hdf5";
  if( argc > 5 ) dumppath = argv[5];
  HighFive::File fdump( dumppath, HighFive::File::Truncate );
  int ipt_dump = 0;

  for(const double MBGeV : MBGeVs){
    // for(const double MB : std::vector<double>{3.0, 3.3, 3.6, 3.9}){
    for(const double MB : MBs){
      // for(double MB=2.6; MB<=4.2; MB+=0.1){
      // for(const double MB : std::vector<double>{5.0}){
      const double sqrt_t0_inv_GeV = MBGeV/MB; // # GeV; free parameter
      const double MP = MP_GeV / sqrt_t0_inv_GeV;

      // # vary m=0.2
      std::vector<std::vector<GWParams>> jk_fO_m0p2000;
      std::string mass="0p2000";

      for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
        std::vector<GWParams> jk(nbins);
# ifdef is_eps_trick
        const Vk jk_Tcparams = eps_trick( jk_Tcparams_m0p2000[jdrop], eps);
        const Vk jk_DeltaVparams = eps_trick( jk_DeltaVparams_m0p2000[jdrop], eps );
        const Vk jk_S3params = eps_trick( jk_S3params_m0p2000[jdrop], eps );
#else
        const Vk& jk_Tcparams = jk_Tcparams_m0p2000[jdrop];
        const Vk& jk_DeltaVparams = jk_DeltaVparams_m0p2000[jdrop];
        const Vk& jk_S3params = jk_S3params_m0p2000[jdrop];
#endif
        // parallel over jackknife bins: each iteration writes its own jk[ibin] with no
        // cross-iteration dependency, so results are bitwise-identical to the serial loop.
#ifdef _OPENMP
// #pragma omp parallel for schedule(dynamic) num_threads(2) // NcRedo: uniform 2 threads/process (superseded for the dump run)
#pragma omp parallel for schedule(dynamic) // dump run: honor OMP_NUM_THREADS from the env so the launcher caps total threads = NPAR x OMP_NUM_THREADS; numerics are bitwise-identical to serial (see note above)
#endif
        for(int ibin=0; ibin<nbins; ibin++){
          F_Tc f_Tc( jk_Tcparams[ibin] );
          F_DeltaV f_DV( jk_DeltaVparams[ibin] );
          F_S3hat f_S3( jk_S3params[ibin] );

          double alpha, vJ, beta_tilde, Tstar, Tc;
          get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
                         f_Tc, f_DV, f_S3,
                         MB, MP, sqrt_t0_inv_GeV, NTs, threshold2 );
          assert(Tstar-Tc > threshold);

          double f_m, om_m;
          get_freq_amplitude( f_m, om_m,
                              alpha, vJ, beta_tilde, Tstar, // Tc,
                              sqrt_t0_inv_GeV);
          jk[ibin] <<
            f_m, om_m,
            alpha, beta_tilde,
            Tstar, Tc,
            MB/Tc, (Tc-Tstar);
        }
        jk_fO_m0p2000.push_back( jk );
      }

      // # vary m=0.3
      std::vector<std::vector<GWParams>> jk_fO_m0p3000;
      mass="0p3000";

      for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
        std::vector<GWParams> jk(nbins);
        // std::vector<GWParams> jk(nbins, GWParams::Zero(nops));
#ifdef is_eps_trick
        const Vk jk_Tcparams = eps_trick( jk_Tcparams_m0p3000[jdrop], eps );
        const Vk jk_DeltaVparams = eps_trick( jk_DeltaVparams_m0p3000[jdrop], eps );
        const Vk jk_S3params = eps_trick( jk_S3params_m0p3000[jdrop], eps );
#else
        const Vk& jk_Tcparams = jk_Tcparams_m0p3000[jdrop];
        const Vk& jk_DeltaVparams = jk_DeltaVparams_m0p3000[jdrop];
        const Vk& jk_S3params = jk_S3params_m0p3000[jdrop];
#endif

        // parallel over jackknife bins: each iteration writes its own jk[ibin] with no
        // cross-iteration dependency, so results are bitwise-identical to the serial loop.
#ifdef _OPENMP
// #pragma omp parallel for schedule(dynamic) num_threads(2) // NcRedo: uniform 2 threads/process (superseded for the dump run)
#pragma omp parallel for schedule(dynamic) // dump run: honor OMP_NUM_THREADS from the env so the launcher caps total threads = NPAR x OMP_NUM_THREADS; numerics are bitwise-identical to serial (see note above)
#endif
        for(int ibin=0; ibin<nbins; ibin++){
          F_Tc f_Tc( jk_Tcparams[ibin] );
          F_DeltaV f_DV( jk_DeltaVparams[ibin] );
          F_S3hat f_S3( jk_S3params[ibin] );
        // for(int ibin=0; ibin<nbins; ibin++){
        //   F_Tc f_Tc( jk_Tcparams_m0p3000[jdrop][ibin] );
        //   F_DeltaV f_DV( jk_DeltaVparams_m0p3000[jdrop][ibin] );
        //   F_S3hat f_S3( jk_S3params_m0p3000[jdrop][ibin] );

          double alpha, vJ, beta_tilde, Tstar, Tc;
          get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
                         f_Tc, f_DV, f_S3,
                         MB, MP, sqrt_t0_inv_GeV, NTs, threshold2 );
          assert(Tstar-Tc > threshold);

          double f_m, om_m;
          get_freq_amplitude( f_m, om_m,
                              alpha, vJ, beta_tilde, Tstar, // Tc,
                              sqrt_t0_inv_GeV);
          // jk[ibin] << f_m, om_m, alpha, beta_tilde, Tstar, Tc, Tc-Tstar;
          jk[ibin] <<
            f_m, om_m,
            alpha, beta_tilde,
            Tstar, Tc,
            MB/Tc, (Tc-Tstar);
          // jk[ibin] << f_m, om_m;
        }
        jk_fO_m0p3000.push_back( jk );
      }

      // # vary m=0.4
      std::vector<std::vector<GWParams>> jk_fO_m0p4000;
      mass="0p4000";

      for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
        std::vector<GWParams> jk(nbins);
        // std::vector<GWParams> jk(nbins, GWParams::Zero(nops));
        // for(int ibin=0; ibin<nbins; ibin++){
        //   F_Tc f_Tc( jk_Tcparams_m0p4000[jdrop][ibin] );
        //   F_DeltaV f_DV( jk_DeltaVparams_m0p4000[jdrop][ibin] );
        //   F_S3hat f_S3( jk_S3params_m0p4000[jdrop][ibin] );
#ifdef is_eps_trick
        const Vk jk_Tcparams = eps_trick( jk_Tcparams_m0p4000[jdrop], eps );
        const Vk jk_DeltaVparams = eps_trick( jk_DeltaVparams_m0p4000[jdrop], eps );
        const Vk jk_S3params = eps_trick( jk_S3params_m0p4000[jdrop], eps );
#else
        const Vk& jk_Tcparams = jk_Tcparams_m0p4000[jdrop];
        const Vk& jk_DeltaVparams = jk_DeltaVparams_m0p4000[jdrop];
        const Vk& jk_S3params = jk_S3params_m0p4000[jdrop];
#endif

        // parallel over jackknife bins: each iteration writes its own jk[ibin] with no
        // cross-iteration dependency, so results are bitwise-identical to the serial loop.
#ifdef _OPENMP
// #pragma omp parallel for schedule(dynamic) num_threads(2) // NcRedo: uniform 2 threads/process (superseded for the dump run)
#pragma omp parallel for schedule(dynamic) // dump run: honor OMP_NUM_THREADS from the env so the launcher caps total threads = NPAR x OMP_NUM_THREADS; numerics are bitwise-identical to serial (see note above)
#endif
        for(int ibin=0; ibin<nbins; ibin++){
          F_Tc f_Tc( jk_Tcparams[ibin] );
          F_DeltaV f_DV( jk_DeltaVparams[ibin] );
          F_S3hat f_S3( jk_S3params[ibin] );

          double alpha, vJ, beta_tilde, Tstar, Tc;
          get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
                         f_Tc, f_DV, f_S3,
                         MB, MP, sqrt_t0_inv_GeV, NTs, threshold2 );
          assert(Tstar-Tc > threshold);

          double f_m, om_m;
          get_freq_amplitude( f_m, om_m,
                              alpha, vJ, beta_tilde, Tstar, // Tc,
                              sqrt_t0_inv_GeV);
          // jk[ibin] << f_m, om_m;
          // jk[ibin] << f_m, om_m, alpha, beta_tilde, Tstar, Tc, Tc-Tstar;
          jk[ibin] <<
            f_m, om_m,
            alpha, beta_tilde,
            Tstar, Tc,
            MB/Tc, (Tc-Tstar);
        }
        jk_fO_m0p4000.push_back( jk );
      }


      GWParams var_tot = GWParams::Zero();

      {
        // # vary m=0.2
        mass="0p2000";
        for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
          JackknifeSimp<GWParams> jk( nbins );
          jk.jack_avg = jk_fO_m0p2000[jdrop];
          jk.finalize();
          var_tot += jk.var;
        }
      }
      {
        // # vary m=0.3
        mass="0p3000";
        for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
          JackknifeSimp<GWParams> jk( nbins );
          jk.jack_avg = jk_fO_m0p3000[jdrop];
          jk.finalize();
          var_tot += jk.var;
        }
      }
      {
        // # vary m=0.4
        mass="0p4000";
        for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
          JackknifeSimp<GWParams> jk( nbins );
          jk.jack_avg = jk_fO_m0p4000[jdrop];
          jk.finalize();
          var_tot += jk.var;
        }
      }

      // #######################MEAN

      F_Tc f_Tc( Tcparams );
      F_DeltaV f_DV( DeltaVparams );
      F_S3hat f_S3( S3params );

      double alpha, vJ, beta_tilde, Tstar, Tc;
      get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
                     f_Tc, f_DV, f_S3,
                     MB, MP, sqrt_t0_inv_GeV, NTs );
      assert(Tstar-Tc > threshold);

      double f_m, om_m;
      get_freq_amplitude( f_m, om_m,
                          alpha, vJ, beta_tilde, Tstar, // Tc,
                          sqrt_t0_inv_GeV);

      // per-jk dump (claude): write every jk sample (8-vec GWParams) for all three masses,
      // plus the mean 8-vec, into a group for this (MBGeV, MB) point. Ordering matches the
      // jk samples: (f, Omega, alpha, beta_tilde, Tstar, Tc, MB/Tc, Tc-Tstar).
      {
        HighFive::Group g = fdump.createGroup( "pt_" + std::to_string(ipt_dump) );
        g.createDataSet( "MBGeV", MBGeV );
        g.createDataSet( "MB", MB );
        g.createDataSet( "sqrt_t0_inv_GeV", sqrt_t0_inv_GeV );

        V meanvec(8);
        meanvec[0] = f_m;
        meanvec[1] = om_m;
        meanvec[2] = alpha;
        meanvec[3] = beta_tilde;
        meanvec[4] = Tstar;
        meanvec[5] = Tc;
        meanvec[6] = MB/Tc;
        meanvec[7] = Tc-Tstar;
        g.createDataSet( "mean", meanvec );

        const Vjk_ dump_m0p2000 = to_nested_gwparams( jk_fO_m0p2000 );
        const Vjk_ dump_m0p3000 = to_nested_gwparams( jk_fO_m0p3000 );
        const Vjk_ dump_m0p4000 = to_nested_gwparams( jk_fO_m0p4000 );
        g.createDataSet( "jk_m0p2000", dump_m0p2000 );
        g.createDataSet( "jk_m0p3000", dump_m0p3000 );
        g.createDataSet( "jk_m0p4000", dump_m0p4000 );

        ipt_dump++;
      }

      std::cout << MBGeV << " " << MB << " " << sqrt_t0_inv_GeV << " " // 0, 1, 2
                << f_m << " " << om_m << " " // 3, 4
                << std::sqrt( var_tot[0] )/epsilon/eps << " " << std::sqrt( var_tot[1] )/epsilon/eps << " " // 5, 6
                << alpha << " " << vJ << " " << beta_tilde << " " // 7, 8, 9
                << std::sqrt( var_tot[2] )/epsilon/eps << " " << std::sqrt( var_tot[3] )/epsilon/eps << " " // 10. 11
                << Tstar << " " << Tc << " " // 12, 13
                << std::sqrt( var_tot[4] )/epsilon/eps << " " // 14  epscorrected /epsilon->/epsilon/eps (Tstar err)
                << std::sqrt( var_tot[5] )/epsilon/eps << " " // 15  epscorrected /epsilon->/epsilon/eps (Tc err)
                << Tstar*sqrt_t0_inv_GeV << " " << Tc*sqrt_t0_inv_GeV << " " // 16. 17
                << std::sqrt( var_tot[4] )/epsilon/eps*sqrt_t0_inv_GeV << " " // 18
                << std::sqrt( var_tot[5] )/epsilon/eps*sqrt_t0_inv_GeV << " " // 19
                << MB/Tc << " " // 20
                << std::sqrt( var_tot[6] )/epsilon/eps << " " // 21
                << (Tc-Tstar) << " " // 22
                << std::sqrt( var_tot[7] )/epsilon/eps << " " // 23
                << std::endl;
    }}
  return 0;
}



      // // # vary m=0.3
      // std::vector<GWParams> jk_fO_m0p3000;
      // mass="0p3000";

      // for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
      //   GWParams jk_;
      //   for(int ibin=0; ibin<nbins; ibin++){
      //     // # print(jdrop, ibin)
      //     F_Tc f_Tc( jk_Tcparams_m0p3000[jdrop][ibin] );
      //     F_DeltaV f_DV( jk_DeltaVparams_m0p3000[jdrop][ibin] );
      //     F_S3hat f_S3( jk_S3params_m0p3000[jdrop][ibin] );
      //     // std::cout << "debug. f_Tc = " << f_Tc( 3.2) << std::endl;
      //     // std::cout << "debug. f_DV = " << f_DV(-0.000152, 3.2) << std::endl;
      //     // std::cout << "debug. f_S3 = " << f_S3(-0.000152, 3.2) << std::endl;

      //     double alpha, vJ, beta_tilde, Tstar, Tc;
      //     get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
      //                    f_Tc, f_DV, f_S3,
      //                    MB, MP, NTs );
      //     // std::cout << alpha << " "
      //     //           << vJ << " "
      //     //           << beta_tilde << " "
      //     //           << Tstar << " "
      //     //           << Tc << std::endl;
      //     // std::cout << Tstar-Tc << std::endl;
      //     assert(Tstar-Tc > -0.00035);
      //     // meanparams = [alpha, vJ, beta_tilde, Tstar, Tc]

      //     double f_m, om_m;
      //     get_freq_amplitude( f_m, om_m,
      //                         alpha, vJ, beta_tilde, Tstar, // Tc,
      //                         sqrt_t0_inv_GeV);
      //     jk_ << f_m, om_m;
      //     // return 1;
      //   }
      //   jk_fO_m0p3000.push_back( jk_ );
      // }

      // // # vary m=0.4
      // std::vector<GWParams> jk_fO_m0p4000;
      // mass="0p4000";

      // for(int jdrop=0; jdrop<nbetas[mass]; jdrop++){
      //   GWParams jk_;
      //   for(int ibin=0; ibin<nbins; ibin++){
      //     // # print(jdrop, ibin)
      //     F_Tc f_Tc( jk_Tcparams_m0p4000[jdrop][ibin] );
      //     F_DeltaV f_DV( jk_DeltaVparams_m0p4000[jdrop][ibin] );
      //     F_S3hat f_S3( jk_S3params_m0p4000[jdrop][ibin] );
      //     // std::cout << "debug. f_Tc = " << f_Tc( 3.2) << std::endl;
      //     // std::cout << "debug. f_DV = " << f_DV(-0.000152, 3.2) << std::endl;
      //     // std::cout << "debug. f_S3 = " << f_S3(-0.000152, 3.2) << std::endl;

      //     double alpha, vJ, beta_tilde, Tstar, Tc;
      //     get_gw_params( alpha, vJ, beta_tilde, Tstar, Tc,
      //                    f_Tc, f_DV, f_S3,
      //                    MB, MP, NTs );
      //     // std::cout << alpha << " "
      //     //           << vJ << " "
      //     //           << beta_tilde << " "
      //     //           << Tstar << " "
      //     //           << Tc << std::endl;
      //     // std::cout << Tstar-Tc << std::endl;
      //     assert(Tstar-Tc > -0.00035);
      //     // meanparams = [alpha, vJ, beta_tilde, Tstar, Tc]

      //     double f_m, om_m;
      //     get_freq_amplitude( f_m, om_m,
      //                         alpha, vJ, beta_tilde, Tstar, // Tc,
      //                         sqrt_t0_inv_GeV);
      //     jk_ << f_m, om_m;
      //     // return 1;
      //   }
      //   jk_fO_m0p4000.push_back( jk_ );
      // }
